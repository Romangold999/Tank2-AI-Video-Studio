package com.romangold.tank2aivideostudio

import android.app.Activity
import android.os.Bundle
import android.graphics.Color
import android.graphics.Typeface
import android.content.Intent
import android.net.Uri
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.*

class MainActivity : Activity() {
    private val bg = Color.rgb(15, 18, 24)
    private val panel = Color.rgb(26, 31, 40)
    private val muted = Color.rgb(157, 168, 184)
    private val accent = Color.rgb(112, 160, 255)
    private lateinit var imagePreview: ImageView
    private lateinit var imageHint: TextView
    private lateinit var durationValue: TextView
    private lateinit var progress: ProgressBar
    private lateinit var progressText: TextView
    private lateinit var timeText: TextView
    private lateinit var videoPreview: VideoView
    private lateinit var downloadButton: Button
    private var selectedImage: Uri? = null

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
        content.addView(label("Локальная генерация видео из изображения", 13f, muted).top(5))
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

        content.addView(sectionTitle("ДЛИТЕЛЬНОСТЬ").top(22))
        val durationRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        durationRow.addView(label("Целевая длина ролика", 14f, muted), LinearLayout.LayoutParams(0, -2, 1f))
        durationValue = label("2 сек", 16f, Color.WHITE, true)
        durationRow.addView(durationValue)
        content.addView(durationRow.top(8))
        val duration = SeekBar(this).apply {
            max = 8
            progress = 0
            progressTintList = android.content.res.ColorStateList.valueOf(accent)
            thumbTintList = android.content.res.ColorStateList.valueOf(accent)
        }
        duration.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, value: Int, fromUser: Boolean) {
                durationValue.text = "${value + 2} сек"
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
        content.addView(duration.top(3))
        content.addView(label("Фактическая длительность будет зависеть от возможностей подключённого режима генерации.", 12f, muted).top(2))

        val generateButton = button("Создать видео")
        content.addView(generateButton.top(22))
        generateButton.setOnClickListener {
            val message = when {
                selectedImage == null -> "Сначала выберите изображение"
                else -> "Нейросетевой движок ещё подключается. Генерация пока недоступна."
            }
            Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        }

        content.addView(sectionTitle("ПРОГРЕСС").top(24))
        progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            progress = 0
            progressTintList = android.content.res.ColorStateList.valueOf(accent)
        }
        content.addView(progress.top(10))
        progressText = label("Ожидание запуска", 14f, Color.WHITE, true)
        content.addView(progressText.top(8))
        timeText = label("Прошло: —    Осталось: —", 12f, muted)
        content.addView(timeText.top(5))
        content.addView(label("Здесь будут отображаться этап, процент выполнения и расчёт оставшегося времени.", 12f, muted).top(8))

        content.addView(sectionTitle("РЕЗУЛЬТАТ").top(24))
        videoPreview = VideoView(this).apply {
            visibility = View.GONE
            setBackgroundColor(Color.BLACK)
        }
        content.addView(videoPreview.top(10))
        downloadButton = button("Скачать видео").apply { isEnabled = false; alpha = 0.45f }
        content.addView(downloadButton.top(10))
        downloadButton.setOnClickListener {
            Toast.makeText(this, "Сначала нужно завершить генерацию видео", Toast.LENGTH_SHORT).show()
        }
        content.addView(label("Генерация будет выполняться на телефоне. Исходное изображение не отправляется в облако.", 12f, muted).top(22))
        scroll.addView(content)
        root.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        setContentView(root)
    }

    @Deprecated("Uses the platform activity result callback for broad Android compatibility")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PICK_IMAGE && resultCode == RESULT_OK) {
            data?.data?.let { uri ->
                selectedImage = uri
                imagePreview.setImageURI(uri)
                imagePreview.visibility = View.VISIBLE
                imageHint.visibility = View.GONE
            }
        }
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

    companion object { private const val PICK_IMAGE = 4102 }
}
