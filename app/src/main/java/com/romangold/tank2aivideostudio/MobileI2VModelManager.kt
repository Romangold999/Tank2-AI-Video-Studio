package com.romangold.tank2aivideostudio

import android.content.Context
import java.io.File
import java.security.MessageDigest
import org.json.JSONObject

/**
 * Installs the model files bundled in the APK into app-private storage.
 * No network connection is used by this class.
 */
class MobileI2VModelManager(context: Context) {
    private val appContext = context.applicationContext
    private val modelDirectory = File(appContext.filesDir, "models/mobilei2v-720p")

    data class ModelFiles(
        val encoder: File,
        val denoiser: File,
        val decoder: File
    )

    fun ensureModels(
        onProgress: (stage: String, copied: Long, total: Long) -> Unit = { _, _, _ -> }
    ): ModelFiles {
        modelDirectory.mkdirs()
        val manifestAsset = "models/models-manifest.json"
        val manifestFile = File(modelDirectory, "models-manifest.json")
        appContext.assets.open(manifestAsset).use { input ->
            manifestFile.outputStream().buffered().use { output -> input.copyTo(output) }
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

            if (!destination.isFile || destination.length() != expectedSize ||
                sha256(destination) != expectedHash
            ) {
                val temporary = File(modelDirectory, "$name.part")
                temporary.delete()
                appContext.assets.open("models/$name").use { input ->
                    temporary.outputStream().buffered().use { output ->
                        val buffer = ByteArray(1024 * 1024)
                        var copied = 0L
                        while (true) {
                            val count = input.read(buffer)
                            if (count < 0) break
                            output.write(buffer, 0, count)
                            copied += count
                            onProgress(name, copied, expectedSize)
                        }
                    }
                }
                check(temporary.length() == expectedSize) {
                    "Bundled model size mismatch for $name: expected $expectedSize, got ${temporary.length()}"
                }
                check(sha256(temporary) == expectedHash) {
                    temporary.delete()
                    "SHA-256 verification failed for bundled model $name"
                }
                if (destination.exists()) destination.delete()
                check(temporary.renameTo(destination)) { "Could not install bundled model $name" }
            } else {
                onProgress(name, expectedSize, expectedSize)
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
        private const val ENCODER = "ltx_open_vae_encoder.onnx"
        private const val DENOISER = "mobiledit_300m.onnx"
        private const val DECODER = "turbo_vaed_ltx_decoder.onnx"
        private val REQUIRED_MODELS = setOf(ENCODER, DENOISER, DECODER)
    }
}
