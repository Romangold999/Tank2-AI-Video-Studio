package com.romangold.tank2aivideostudio

import android.content.Context
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import org.json.JSONObject

/**
 * Downloads the three MobileI2V ONNX stages once and verifies every byte
 * against the published release manifest before making a model visible.
 *
 * Generation remains fully offline after this one-time download.
 */
class MobileI2VModelManager(context: Context) {
    private val modelDirectory = File(context.applicationContext.filesDir, "models/mobilei2v-720p")

    data class ModelFiles(
        val encoder: File,
        val denoiser: File,
        val decoder: File
    )

    fun ensureModels(
        onProgress: (stage: String, downloaded: Long, total: Long) -> Unit = { _, _, _ -> }
    ): ModelFiles {
        modelDirectory.mkdirs()
        val manifestFile = File(modelDirectory, "models-manifest.json")
        downloadVerified(MANIFEST_URL, manifestFile, null) { done, total ->
            onProgress("manifest", done, total)
        }
        val manifest = JSONObject(manifestFile.readText())
        val models = manifest.getJSONArray("models")
        val files = linkedMapOf<String, File>()
        for (index in 0 until models.length()) {
            val item = models.getJSONObject(index)
            val name = item.getString("file")
            require(name in REQUIRED_MODELS) { "Unexpected model in manifest: $name" }
            val expectedSize = item.getLong("size_bytes")
            val expectedHash = item.getString("sha256")
            val destination = File(modelDirectory, name)
            val url = "$RELEASE_BASE/$name"
            downloadVerified(url, destination, expectedHash) { done, total ->
                onProgress(name, done, if (total > 0L) total else expectedSize)
            }
            require(destination.length() == expectedSize) {
                "Model size mismatch for $name: expected $expectedSize, got ${destination.length()}"
            }
            files[name] = destination
        }
        val missing = REQUIRED_MODELS - files.keys
        require(missing.isEmpty()) { "Model manifest is incomplete: $missing" }
        return ModelFiles(
            encoder = files.getValue(ENCODER),
            denoiser = files.getValue(DENOISER),
            decoder = files.getValue(DECODER)
        )
    }

    private fun downloadVerified(
        sourceUrl: String,
        destination: File,
        expectedSha256: String?,
        onProgress: (Long, Long) -> Unit
    ) {
        if (destination.isFile && destination.length() > 0L &&
            (expectedSha256 == null || sha256(destination) == expectedSha256)
        ) return

        val temporary = File(destination.parentFile, destination.name + ".part")
        temporary.delete()
        val connection = (URL(sourceUrl).openConnection() as HttpURLConnection).apply {
            connectTimeout = 30_000
            readTimeout = 120_000
            instanceFollowRedirects = true
            requestMethod = "GET"
        }
        try {
            connection.connect()
            check(connection.responseCode in 200..299) {
                "Model download failed (HTTP ${connection.responseCode}): $sourceUrl"
            }
            val total = connection.contentLengthLong
            connection.inputStream.use { input ->
                temporary.outputStream().buffered().use { output ->
                    val buffer = ByteArray(1024 * 1024)
                    var downloaded = 0L
                    while (true) {
                        val count = input.read(buffer)
                        if (count < 0) break
                        output.write(buffer, 0, count)
                        downloaded += count
                        onProgress(downloaded, total)
                    }
                }
            }
        } catch (error: Exception) {
            temporary.delete()
            throw error
        } finally {
            connection.disconnect()
        }

        if (expectedSha256 != null) {
            check(sha256(temporary) == expectedSha256) {
                temporary.delete()
                "SHA-256 verification failed for ${destination.name}"
            }
        }
        if (destination.exists()) destination.delete()
        check(temporary.renameTo(destination)) {
            "Could not install model file ${destination.name}"
        }
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(1024 * 128)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    companion object {
        private const val RELEASE_BASE =
            "https://github.com/Romangold999/Tank2-AI-Video-Studio/releases/download/tank2-models-v0.1.0"
        private const val MANIFEST_URL = "$RELEASE_BASE/models-manifest.json"
        private const val ENCODER = "ltx_open_vae_encoder.onnx"
        private const val DENOISER = "mobiledit_300m.onnx"
        private const val DECODER = "turbo_vaed_ltx_decoder.onnx"
        private val REQUIRED_MODELS = setOf(ENCODER, DENOISER, DECODER)
    }
}
