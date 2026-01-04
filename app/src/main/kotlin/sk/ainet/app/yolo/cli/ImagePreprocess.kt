package sk.ainet.app.yolo.cli

import sk.ainet.context.ExecutionContext
import sk.ainet.io.image.*
import sk.ainet.lang.model.dnn.yolo.YoloInput
import sk.ainet.lang.model.dnn.yolo.YoloPreprocess
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import java.io.File
import javax.imageio.ImageIO
import java.awt.Color
import java.awt.Graphics2D
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import kotlin.math.min

fun createYoloInput(imagePath: String, targetSize: Int, ctx: ExecutionContext): YoloInput {
    // 1. Load actual image using platform API (ImageIO for JVM)
    val file = File(imagePath)
    val originalImage = ImageIO.read(file) ?: error("Failed to load image at $imagePath")

    // 2. Preprocess: Letterbox resize and convert to Tensor
    val preprocessed = preprocessImage(originalImage, targetSize, ctx)

    // 3. Wrap into YoloInput with metadata
    return YoloPreprocess.fromReadyTensor(
        tensor = preprocessed.tensor,
        originalWidth = originalImage.width,
        originalHeight = originalImage.height,
        inputSize = targetSize,
        scale = preprocessed.scale,
        padW = preprocessed.padX,
        padH = preprocessed.padY
    )
}

data class PreprocessedImage(
    val tensor: Tensor<FP32, Float>,
    val scale: Float,
    val padX: Int,
    val padY: Int
)

fun preprocessImage(image: BufferedImage, targetSize: Int, ctx: ExecutionContext): PreprocessedImage {
    // Letterbox resizing logic
    val scale = min(targetSize.toFloat() / image.width, targetSize.toFloat() / image.height)
    val newW = (image.width * scale).toInt()
    val newH = (image.height * scale).toInt()
    val padX = (targetSize - newW) / 2
    val padY = (targetSize - newH) / 2

    // Create a new canvas with the target size and draw the original image centered (letterboxed)
    val letterboxed = BufferedImage(targetSize, targetSize, BufferedImage.TYPE_INT_RGB)
    val g: Graphics2D = letterboxed.createGraphics()
    g.color = Color.BLACK
    g.fillRect(0, 0, targetSize, targetSize)
    g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
    g.drawImage(image, padX, padY, newW, newH, null)
    g.dispose()

    // Manually convert BufferedImage to FP32 Tensor (1, 3, H, W) to avoid unimplemented FP16 to FP32 conversion
    val shape = sk.ainet.lang.tensor.Shape(intArrayOf(1, 3, targetSize, targetSize))
    val data = FloatArray(targetSize * targetSize * 3)
    
    for (y in 0 until targetSize) {
        for (x in 0 until targetSize) {
            val rgb = letterboxed.getRGB(x, y)
            val r = ((rgb shr 16) and 0xFF) / 255.0f
            val gVal = ((rgb shr 8) and 0xFF) / 255.0f
            val b = (rgb and 0xFF) / 255.0f
            
            // NCHW format
            data[0 * targetSize * targetSize + y * targetSize + x] = r
            data[1 * targetSize * targetSize + y * targetSize + x] = gVal
            data[2 * targetSize * targetSize + y * targetSize + x] = b
        }
    }
    
    val normalizedTensor = ctx.fromFloatArray<FP32, Float>(shape, FP32::class, data)

    return PreprocessedImage(normalizedTensor, scale, padX, padY)
}