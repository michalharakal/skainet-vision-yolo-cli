package sk.ainet.app.yolo.cli

import kotlinx.coroutines.runBlocking
import sk.ainet.lang.model.dnn.yolo.Yolo8
import sk.ainet.lang.model.dnn.yolo.YoloConfig
import sk.ainet.context.DirectCpuExecutionContext

fun main(args: Array<String>) = runBlocking {
    if (args.isEmpty()) {
        println("Usage: <image_path>")
        return@runBlocking
    }

    val ctx = DirectCpuExecutionContext()
    val yolo = Yolo8(YoloConfig())
    val model = yolo.create(ctx)

    // Load and preprocess image (returns a fully configured YoloInput)
    val yoloInput = createYoloInput(args[0], 640, ctx)

    // Run inference inside the coroutine scope
    val detections = yolo.infer(model, yoloInput, ctx)

    // Output results
    detections.forEach { detection ->
        println("${detection.label}: ${detection.score} at ${detection.box}")
    }
}