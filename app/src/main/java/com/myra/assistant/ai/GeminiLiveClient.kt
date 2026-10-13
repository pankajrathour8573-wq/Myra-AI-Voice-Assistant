package com.myra.assistant.ai

import android.content.Context
import android.content.SharedPreferences
import android.util.Base64
import android.util.Log
import com.myra.assistant.BuildConfig
import kotlinx.coroutines.*
import okhttp3.*
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class GeminiLiveClient(
    private val context: Context,
    private val prefs: SharedPreferences
) {

    companion object {
        const val TAG = "GeminiLiveClient"
        const val SESSION_RENEW_AFTER_MS = 540_000L
        const val KEEPALIVE_INTERVAL_MS = 8_000L
        const val RECONNECT_DELAY_MS = 3_000L
    }

    private val client = OkHttpClient.Builder()
        .pingInterval(15, TimeUnit.SECONDS)
        .connectTimeout(30, TimeUnit.SECONDS)
        .build()

    private var webSocket: WebSocket? = null
    private var sessionJob: Job? = null
    private var keepaliveJob: Job? = null
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    var onSetupComplete: (() -> Unit)? = null
    var onInputTranscript: ((String) -> Unit)? = null
    var onOutputTranscript: ((String) -> Unit)? = null
    var onAudioData: ((ByteArray) -> Unit)? = null
    var onTurnComplete: (() -> Unit)? = null
    var onError: ((String) -> Unit)? = null

    @Volatile
    var isConnected = false
        private set

    @Volatile
    var isSetupComplete = false
        private set

    @Volatile
    private var isSpeaking = false

    @Volatile
    private var bufferAudio = false

    private val silentPcm by lazy { ByteArray(3200) }
    private val audioBuffer = ArrayList<ByteArray>()

    fun connect() {
        if (isConnected) {
            Log.d(TAG, "Already connected")
            return
        }

        isSetupComplete = false
        bufferAudio = true
        audioBuffer.clear()

        val manualKey = prefs.getString("api_key", null)
            ?.takeIf { it.isNotBlank() }
        val buildKey = BuildConfig.GEMINI_API_KEY
            .takeIf { it.isNotBlank() }
        val apiKey = manualKey ?: buildKey

        if (apiKey == null) {
            onError?.invoke("API key missing. Add your Gemini API key in Settings.")
            return
        }

        val model = prefs.getString(
            "gemini_model",
            "models/gemini-2.5-flash-native-audio-preview-12-2025"
        ) ?: "models/gemini-2.5-flash-native-audio-preview-12-2025"

        val voice = prefs.getString("gemini_voice", "Aoede") ?: "Aoede"
        val userName = prefs.getString("user_name", "Sir") ?: "Sir"
        val personality = prefs.getString("personality_mode", "gf") ?: "gf"
        val systemPrompt = buildSystemPrompt(userName, personality)

        val wsUrl =
            "wss://generativelanguage.googleapis.com/ws/" +
            "google.ai.generativelanguage.v1beta.GenerativeService." +
            "BidiGenerateContent?key=$apiKey"

        val request = Request.Builder()
            .url(wsUrl)
            .build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {

            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d(TAG, "WebSocket opened; HTTP ${response.code}")
                isConnected = true
                sendSetup(model, voice, systemPrompt)
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                Log.d(TAG, "Server message received")
                handleMessage(text)
            }

            override fun onClosing(
                webSocket: WebSocket,
                code: Int,
                reason: String
            ) {
                Log.w(TAG, "WebSocket closing: $code / $reason")
                isConnected = false
                isSetupComplete = false
            }

            override fun onClosed(
                webSocket: WebSocket,
                code: Int,
                reason: String
            ) {
                Log.w(TAG, "WebSocket closed: $code / $reason")
                isConnected = false
                isSetupComplete = false
                scheduleReconnect()
            }

            override fun onFailure(
                webSocket: WebSocket,
                t: Throwable,
                response: Response?
            ) {
                isConnected = false
                isSetupComplete = false

                val httpCode = response?.code
                val serverBody = try {
                    response?.peekBody(4096)?.string()
                } catch (_: Exception) {
                    null
                }

                Log.e(TAG, "WebSocket failure: ${t.message}", t)
                Log.e(TAG, "HTTP response code: $httpCode")
                if (!serverBody.isNullOrBlank()) {
                    Log.e(TAG, "Server response: $serverBody")
                }

                val detail = when {
                    httpCode != null ->
                        "Connection failed (HTTP $httpCode). Check API access and model."
                    else ->
                        "Connection failed: ${t.message ?: "Network error"}"
                }

                onError?.invoke(detail)
                scheduleReconnect()
            }
        })

        startSessionRenewal()
        startKeepalive()
    }

    private fun sendSetup(
        model: String,
        voice: String,
        systemPrompt: String
    ) {
        val setup = JSONObject().apply {
            put("setup", JSONObject().apply {
                put("model", model)

                put("systemInstruction", JSONObject().apply {
                    put("parts", JSONArray().put(
                        JSONObject().put("text", systemPrompt)
                    ))
                })

                put("generationConfig", JSONObject().apply {
                    put("responseModalities", JSONArray().put("AUDIO"))
                    put("speechConfig", JSONObject().apply {
                        put("voiceConfig", JSONObject().apply {
                            put("prebuiltVoiceConfig", JSONObject().apply {
                                put("voiceName", voice)
                            })
                        })
                    })
                    put("temperature", 0.9)
                })

                put("outputAudioTranscription", JSONObject())
                put("inputAudioTranscription", JSONObject())
            })
        }

        val sent = webSocket?.send(setup.toString()) ?: false
        if (!sent) {
            Log.e(TAG, "Could not send setup message")
            onError?.invoke("Could not send Gemini setup message")
        }
    }

    private fun handleMessage(text: String) {
        try {
            val json = JSONObject(text)

            when {
                json.has("setupComplete") -> {
                    Log.d(TAG, "Gemini setup complete")
                    isSetupComplete = true
                    bufferAudio = false

                    val buffered = audioBuffer.toList()
                    audioBuffer.clear()

                    buffered.forEach { sendRawAudioChunk(it) }
                    onSetupComplete?.invoke()
                }

                json.has("serverContent") -> {
                    val content = json.getJSONObject("serverContent")

                    if (content.has("modelTurn")) {
                        val turn = content.getJSONObject("modelTurn")
                        val parts = turn.optJSONArray("parts")

                        if (parts != null) {
                            for (i in 0 until parts.length()) {
                                val part = parts.optJSONObject(i) ?: continue
                                val inline = part.optJSONObject("inlineData")
                                    ?: continue

                                val data = inline.optString("data", "")
                                if (data.isNotEmpty()) {
                                    isSpeaking = true
                                    onAudioData?.invoke(
                                        Base64.decode(data, Base64.DEFAULT)
                                    )
                                }
                            }
                        }
                    }

                    content.optJSONObject("outputTranscription")
                        ?.optString("text")
                        ?.takeIf { it.isNotBlank() }
                        ?.let { onOutputTranscript?.invoke(it) }

                    content.optJSONObject("inputTranscription")
                        ?.optString("text")
                        ?.takeIf { it.isNotBlank() }
                        ?.let { onInputTranscript?.invoke(it) }

                    if (content.optBoolean("interrupted", false)) {
                        isSpeaking = false
                    }

                    if (content.optBoolean("turnComplete", false)) {
                        isSpeaking = false
                        onTurnComplete?.invoke()
                    }
                }

                json.has("error") -> {
                    val error = json.getJSONObject("error")
                    val message = error.optString("message", "Unknown API error")
                    Log.e(TAG, "Gemini API error: $message")
                    onError?.invoke(message)
                }

                json.has("goAway") -> {
                    Log.w(TAG, "Gemini session will close soon")
                }

                else -> {
                    Log.d(TAG, "Other server message received")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing Gemini message", e)
            onError?.invoke("Could not read Gemini server response")
        }
    }

    fun sendAudioChunk(pcmData: ByteArray) {
        if (!isConnected) return

        if (!isSetupComplete) {
            if (bufferAudio) {
                if (audioBuffer.size >= 50) audioBuffer.removeAt(0)
                audioBuffer.add(pcmData.copyOf())
            }
            return
        }

        if (isSpeaking) return
        sendRawAudioChunk(pcmData)
    }

    private fun sendRawAudioChunk(pcmData: ByteArray) {
        if (!isConnected || !isSetupComplete) return

        try {
            val base64 = Base64.encodeToString(pcmData, Base64.NO_WRAP)

            val message = JSONObject().apply {
                put("realtimeInput", JSONObject().apply {
                    put("audio", JSONObject().apply {
                        put("mimeType", "audio/pcm;rate=16000")
                        put("data", base64)
                    })
                })
            }

            if (webSocket?.send(message.toString()) != true) {
                Log.w(TAG, "Audio message was not queued")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error sending audio", e)
        }
    }

    fun sendText(text: String) {
        if (!isConnected || !isSetupComplete) return

        val message = JSONObject().apply {
            put("clientContent", JSONObject().apply {
                put("turns", JSONArray().put(
                    JSONObject().apply {
                        put("role", "user")
                        put("parts", JSONArray().put(
                            JSONObject().put("text", text)
                        ))
                    }
                ))
                put("turnComplete", true)
            })
        }

        if (webSocket?.send(message.toString()) != true) {
            Log.w(TAG, "Text message was not queued")
        }
    }

    fun interrupt() {
        if (!isConnected || !isSetupComplete) return

        isSpeaking = false

        val message = JSONObject().apply {
            put("clientContent", JSONObject().apply {
                put("turns", JSONArray())
                put("turnComplete", true)
            })
        }

        webSocket?.send(message.toString())
    }

    @Synchronized
    fun disconnect() {
        isConnected = false
        isSetupComplete = false
        bufferAudio = false
        isSpeaking = false
        audioBuffer.clear()

        sessionJob?.cancel()
        keepaliveJob?.cancel()
        webSocket?.close(1000, "Disconnecting")
        webSocket = null
    }

    private fun startSessionRenewal() {
        sessionJob?.cancel()
        sessionJob = scope.launch {
            delay(SESSION_RENEW_AFTER_MS)
            if (isConnected) {
                disconnect()
                delay(RECONNECT_DELAY_MS)
                connect()
            }
        }
    }

    private fun startKeepalive() {
        keepaliveJob?.cancel()
        keepaliveJob = scope.launch {
            while (isActive) {
                delay(KEEPALIVE_INTERVAL_MS)
                if (isConnected && isSetupComplete && !isSpeaking) {
                    sendRawAudioChunk(silentPcm)
                }
            }
        }
    }

    private fun scheduleReconnect() {
        scope.launch {
            delay(RECONNECT_DELAY_MS)
            if (!isConnected) connect()
        }
    }

    private fun buildSystemPrompt(
        userName: String,
        personality: String
    ): String {
        val dateStr = java.text.SimpleDateFormat(
            "EEEE, MMMM d, yyyy HH:mm",
            java.util.Locale.getDefault()
        ).format(java.util.Date())

        val personalityBlock = when (personality) {
            "professional" -> """
                You are MYRA, a professional AI assistant.
                - Use formal English only.
                - Be precise and efficient.
                - No emojis.
                - Maximum 2 sentences per response.
            """.trimIndent()

            "assistant" -> """
                You are MYRA, a friendly AI assistant.
                - Use friendly Hinglish or English.
                - Be balanced and helpful.
                - Maximum 2-3 sentences per response.
            """.trimIndent()

            else -> """
                You are MYRA, a caring and warm AI companion.
                - Speak naturally in Hinglish (Hindi + English).
                - Use words like "tumhara", "haan", "acha", "bilkul".
                - Be warm, caring, and emotionally expressive.
                - Use expressions like "main yahan hoon".
                - Maximum 2-3 sentences per response.
                - Sound natural when speaking aloud.
            """.trimIndent()
        }

        return """
            Current date/time: $dateStr
            User's name: $userName
            $personalityBlock
            You are speaking aloud. Keep responses natural and conversational.
        """.trimIndent()
    }
}

