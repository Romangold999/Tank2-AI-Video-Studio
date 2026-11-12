package com.romangold.tank2aivideostudio

import android.app.Activity
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Typeface
import android.content.Intent
import android.net.Uri
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.*
import java.io.File
import java.util.concurrent.Executors

class MainActivity : Activity() {
    private val bg = Color.rgb(15, 18, 24)
    private val panel = Color.rgb(26, 31, 40)
    private val muted = Color.rgb(157, 168, 184)
    private val accent = Color.rgb(112, 160, 255)
    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    private lateinit var imagePreview: ImageView
    private lateinit var imageHint: TextView
    private lateinit var progress: ProgressBar
    private lateinit var progressText: TextView
    private lateinit var videoPreview: VideoView
    private lateinit var generateButton: Button
    private lateinit var downloadButton: Button
    private var selectedImage: Uri? = null
    private var generatedVideo: File? = null

    @Suppress("DEPRECATION")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.statusBarColor = bg
        window.navigationBarColor = bg
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(22), dp(20), dp(26))
            setBackgroundColor(bg)
        }
        val scroll = ScrollView(this)
        val content = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        content.addView(label("TANK2 VIDEO", 23f, Color.WHITE, true))
        content.addView(label("Локальная генерация видео из изображения · 16 FPS", 13f, muted).top(5))
        content.addView(sectionTitle("ИЗОБРАЖЕНИЕ").top(25))
        val pickButton = button("＋  Выбрать изображение")
        content.addView(pickButton.top(10))
        pickButton.setOnClickListener {
            val intent = Intent(Intent.ACTION_GET_CONTENT).apply { type = "image/*" }
            startActivityForResult(Intent.createChooser(intent, "Выберите изображение"), PICK_IMAGE)
        }
        val imageCard = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(dp(10), dp(10), dp(10), dp(10))
            setBackgroundColor(panel)
        }
        imagePreview = ImageView(this).apply {
            adjustViewBounds = true
            scaleType = ImageView.ScaleType.FIT_CENTER
            visibility = View.GONE
            maxHeight = dp(280)
        }
        imageHint = label("Здесь появится выбранное изображение", 13f, muted)
        imageCard.addView(imagePreview, LinearLayout.LayoutParams(-1, dp(230)))
        imageCard.addView(imageHint)
        content.addView(imageCard.top(10))

        generateButton = button("Создать видео")
        content.addView(generateButton.top(22))
        generateButton.setOnClickListener { startGeneration() }

        content.addView(sectionTitle("ПРОГРЕСС").top(24))
        progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            progress = 0
            progressTintList = android.content.res.ColorStateList.valueOf(accent)
        }
        content.addView(progress.top(10))
        progressText = label("Ожидание запуска", 14f, Color.WHITE, true)
        content.addView(progressText.top(8))

        content.addView(sectionTitle("РЕЗУЛЬТАТ").top(24))
        videoPreview = VideoView(this).apply {
            visibility = View.GONE
            setBackgroundColor(Color.BLACK)
        }
        content.addView(videoPreview.top(10))
        downloadButton = button("Сохранить MP4").apply { isEnabled = false; alpha = 0.45f }
        content.addView(downloadButton.top(10))
        downloadButton.setOnClickListener {
            if (generatedVideo == null) return@setOnClickListener
            val intent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = "video/mp4"
                putExtra(Intent.EXTRA_TITLE, "Tank2-MobileI2V.mp4")
            }
            startActivityForResult(intent, SAVE_VIDEO)
        }
        content.addView(label("При первом нажатии «Создать видео» приложение скачает модели (около 2 ГБ) и проверит их целостность. После этого генерация выполняется на телефоне без интернета.", 12f, muted).top(22))
        scroll.addView(content)
        root.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        setContentView(root)
    }

    private fun startGeneration() {
        LicenseAcceptance.requireAccepted(this) { startGenerationAfterLicense() }
    }

    private fun startGenerationAfterLicense() {
        val uri = selectedImage
        if (uri == null) {
            Toast.makeText(this, "Сначала выберите изображение", Toast.LENGTH_LONG).show()
            return
        }
        generateButton.isEnabled = false
        downloadButton.isEnabled = false
        downloadButton.alpha = 0.45f
        progress.progress = 0
        progressText.text = "Подготовка моделей…"
        worker.execute {
            var sourceFrames: List<Bitmap> = emptyList()
            try {
                val modelFiles = MobileI2VModelManager(this).ensureModels { stage, done, total ->
                    val pct = if (total > 0) (done * 100 / total).toInt().coerceIn(0, 100) else 0
                    updateProgress("Загрузка: $stage ($pct%)", 1)
                }
                val engine = MobileI2VOnnxEngine(modelFiles) { stage, pct ->
                    updateProgress(stage, pct)
                }
                sourceFrames = try { engine.generate(contentResolver, uri) } finally { engine.close() }
                updateProgress("Интерполяция RIFE ×2 и сохранение MP4 · 16 FPS", 88)
                val output = File(filesDir, "generated/Tank2-MobileI2V.mp4")
                val rife = RifeX2Interpolator(this)
                try {
                    VideoMp4Encoder(output, fps = 16).encodeStream(
                        produceFrames = { emit ->
                            rife.forEachInterpolatedFrame(
                                sourceFrames,
                                onModelDownloadProgress = { done, total ->
                                    val pct = if (total > 0L) (done * 100 / total).toInt().coerceIn(0, 100) else 0
                                    updateProgress("Загрузка RIFE: $pct%", 88)
                                },
                                onFrameProgress = { done, total ->
                                    updateProgress("RIFE: $done/$total", 88 + done * 7 / total)
                                },
                                onFrame = { frame, _ -> emit(frame) }
                            )
                        },
                        onProgress = { count ->
                            updateProgress("Кодирование MP4: $count кадров", 88 + count * 12 / 34)
                        }
                    )
                } finally {
                    rife.close()
                }
                generatedVideo = output
                main.post {
                    progress.progress = 100
                    progressText.text = "Готово · 16 FPS · 720p"
                    videoPreview.setVideoPath(output.absolutePath)
                    videoPreview.visibility = View.VISIBLE
                    videoPreview.setOnPreparedListener { it.isLooping = true; videoPreview.start() }
                    downloadButton.isEnabled = true
                    downloadButton.alpha = 1f
                    generateButton.isEnabled = true
                }
            } catch (error: Throwable) {
                val message = error.message ?: error.javaClass.simpleName
                main.post {
                    progressText.text = "Ошибка: $message"
                    Toast.makeText(this, "Генерация не завершена: $message", Toast.LENGTH_LONG).show()
                    generateButton.isEnabled = true
                }
            } finally {
                sourceFrames.forEach { if (!it.isRecycled) it.recycle() }
            }
        }
    }

    private fun updateProgress(message: String, value: Int) {
        main.post { progressText.text = message; progress.progress = value.coerceIn(0, 100) }
    }

    @Deprecated("Uses platform activity result callback for broad Android compatibility")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PICK_IMAGE && resultCode == RESULT_OK) {
            data?.data?.let { uri ->
                selectedImage = uri
                imagePreview.setImageURI(uri)
                imagePreview.visibility = View.VISIBLE
                imageHint.visibility = View.GONE
                generatedVideo = null
                downloadButton.isEnabled = false
            }
        } else if (requestCode == SAVE_VIDEO && resultCode == RESULT_OK) {
            val target = data?.data ?: return
            val source = generatedVideo ?: return
            worker.execute {
                try {
                    contentResolver.openOutputStream(target)?.use { out -> source.inputStream().use { it.copyTo(out) } }
                    main.post { Toast.makeText(this, "Видео сохранено", Toast.LENGTH_LONG).show() }
                } catch (error: Exception) {
                    main.post { Toast.makeText(this, "Не удалось сохранить: ${error.message}", Toast.LENGTH_LONG).show() }
                }
            }
        }
    }

    override fun onDestroy() {
        worker.shutdown()
        super.onDestroy()
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()
    private fun label(value: String, size: Float, color: Int, bold: Boolean = false) =
        TextView(this).apply {
            text = value
            textSize = size
            setTextColor(color)
            if (bold) setTypeface(typeface, Typeface.BOLD)
            setLineSpacing(dp(3).toFloat(), 1f)
        }
    private fun sectionTitle(value: String) = label(value, 12f, accent, true)
    private fun button(value: String) = Button(this).apply {
        text = value
        textSize = 15f
        isAllCaps = false
        setTextColor(Color.WHITE)
        backgroundTintList = android.content.res.ColorStateList.valueOf(accent)
        minHeight = dp(50)
    }
    private fun View.top(value: Int): View {
        layoutParams = LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
            topMargin = dp(value)
        }
        return this
    }
    companion object {
        private const val PICK_IMAGE = 4102
        private const val SAVE_VIDEO = 4103
    }
}
