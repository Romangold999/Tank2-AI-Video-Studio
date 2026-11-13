package com.romangold.tank2aivideostudio

import android.graphics.Bitmap
import android.graphics.Canvas
import android.media.MediaMetadataRetriever
import android.os.SystemClock
import android.util.Log
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class ModelDownloadInstrumentedTest {
    @Test(timeout = 60_000)
    fun mainActivityLaunchesOnAndroidRuntime() {
        ActivityScenario.launch(MainActivity::class.java).use {
            assertTrue("MainActivity must reach RESUMED state", it.state.name == "RESUMED")
        }
    }

    @Test(timeout = 15 * 60_000L)
    fun extractsAllBundledModelsAndRifeWithoutNetwork() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val started = SystemClock.elapsedRealtime()
        val files = MobileI2VModelManager(context).ensureModels { stage, copied, total ->
            Log.i("Tank2EmbeddedTest", "$stage: $copied/$total")
        }
        listOf(files.encoder, files.denoiser, files.decoder).forEach { file ->
            assertTrue("Missing or empty model: ${file.name}", file.isFile && file.length() > 1_000_000L)
            Log.i("Tank2EmbeddedTest", "VERIFIED ${file.name} ${file.length()} bytes")
        }
        val rifeAsset = context.assets.open("models/rife49.onnx")
        val rifeProbe = File(context.cacheDir, "rife-asset-probe.onnx")
        rifeAsset.use { input -> rifeProbe.outputStream().use { output -> input.copyTo(output) } }
        assertTrue("Bundled RIFE asset is missing", rifeProbe.length() > 1_000_000L)
        rifeProbe.delete()
        assertTrue("All models must be stored in app-private storage",
            listOf(files.encoder, files.denoiser, files.decoder).all {
                it.canonicalPath.startsWith(context.filesDir.canonicalPath)
            })
        Log.i("Tank2EmbeddedTest", "MODEL_EXTRACTION_MS=${SystemClock.elapsedRealtime() - started}")
    }

    @Test(timeout = 5 * 60_000L)
    fun rifeAndMp4SmokeTestOnAndroidRuntime() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val first = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888).apply {
            eraseColor(android.graphics.Color.rgb(30, 70, 110))
        }
        val second = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888).apply {
            eraseColor(android.graphics.Color.rgb(220, 140, 50))
        }
        val frames = listOf(first, second)
        val output = File(context.cacheDir, "tank2-smoke-test.mp4")
        val rife = RifeX2Interpolator(context)
        try {
            VideoMp4Encoder(output, width = 64, height = 64, fps = 16, bitrate = 250_000).encodeStream(
                produceFrames = { emit ->
                    rife.forEachInterpolatedFrame(
                        frames,
                        onFrameProgress = { done, total ->
                            Log.i("Tank2EmbeddedTest", "RIFE_SMOKE $done/$total")
                        },
                        onFrame = { frame, _ -> emit(frame) }
                    )
                }
            )
            assertTrue("MP4 output is missing or too small", output.isFile && output.length() > 1_000L)
            val metadata = MediaMetadataRetriever()
            try {
                metadata.setDataSource(output.absolutePath)
                val duration = metadata.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
                assertTrue("MP4 duration is invalid: $duration ms", duration >= 100L)
                Log.i("Tank2EmbeddedTest", "RIFE_MP4_SMOKE_SUCCESS bytes=${output.length()} duration_ms=$duration")
            } finally {
                metadata.release()
            }
        } finally {
            rife.close()
            frames.forEach { if (!it.isRecycled) it.recycle() }
            output.delete()
        }
    }
}
