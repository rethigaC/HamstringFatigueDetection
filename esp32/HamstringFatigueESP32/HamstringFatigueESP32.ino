#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.cc"


#define TENSOR_ARENA_SIZE (1024 * 1024)

uint8_t *tensor_arena = nullptr;

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;

TfLiteTensor *input_tensor = nullptr;
TfLiteTensor *output_tensor = nullptr;

tflite::AllOpsResolver resolver;



void setup()
{
    Serial.begin(115200);
    delay(2000);

    Serial.println();
    Serial.println("========================================");
    Serial.println(" HAMSTRING FATIGUE DETECTION");
    Serial.println(" TCN + SELF-ATTENTION");
    Serial.println(" INT8 TFLITE");
    Serial.println("========================================");


    // --------------------------------------------------------
    // CHECK PSRAM
    // --------------------------------------------------------

    Serial.println();
    Serial.println("Checking PSRAM...");

    if (!psramFound())
    {
        Serial.println("ERROR: PSRAM NOT FOUND!");
        Serial.println("Use ESP32-S3 board with PSRAM.");

        while (1)
        {
            delay(1000);
        }
    }

    Serial.println("PSRAM FOUND");

    Serial.print("Total PSRAM: ");
    Serial.print(ESP.getPsramSize());
    Serial.println(" bytes");

    Serial.print("Free PSRAM: ");
    Serial.print(ESP.getFreePsram());
    Serial.println(" bytes");


    // --------------------------------------------------------
    // ALLOCATE TENSOR ARENA IN PSRAM
    // --------------------------------------------------------

    Serial.println();
    Serial.println("Allocating Tensor Arena...");

    tensor_arena =
        (uint8_t *)ps_malloc(TENSOR_ARENA_SIZE);

    if (tensor_arena == nullptr)
    {
        Serial.println("ERROR: Tensor Arena allocation failed!");

        while (1)
        {
            delay(1000);
        }
    }

    Serial.print("Tensor Arena: ");
    Serial.print(TENSOR_ARENA_SIZE / 1024);
    Serial.println(" KB");


    // --------------------------------------------------------
    // LOAD MODEL
    // --------------------------------------------------------

    Serial.println();
    Serial.println("Loading TFLite model...");

    model = tflite::GetModel(hamstring_fatigue_model);

    if (model == nullptr)
    {
        Serial.println("ERROR: Model loading failed!");

        while (1)
        {
            delay(1000);
        }
    }

    Serial.println("Model loaded successfully.");

    Serial.print("Model schema version: ");
    Serial.println(model->version());


    // --------------------------------------------------------
    // CHECK SCHEMA
    // --------------------------------------------------------

    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        Serial.println("ERROR: TFLite schema mismatch!");

        while (1)
        {
            delay(1000);
        }
    }

    Serial.println("Schema OK");


    // --------------------------------------------------------
    // CREATE INTERPRETER
    // --------------------------------------------------------

    Serial.println();
    Serial.println("Creating interpreter...");

    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        tensor_arena,
        TENSOR_ARENA_SIZE
    );

    interpreter = &static_interpreter;


    // --------------------------------------------------------
    // ALLOCATE TENSORS
    // --------------------------------------------------------

    Serial.println("Allocating tensors...");

    TfLiteStatus status =
        interpreter->AllocateTensors();

    if (status != kTfLiteOk)
    {
        Serial.println("ERROR: AllocateTensors FAILED!");

        while (1)
        {
            delay(1000);
        }
    }

    Serial.println("Tensor allocation SUCCESS");


    // --------------------------------------------------------
    // GET INPUT / OUTPUT
    // --------------------------------------------------------

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);


    if (input_tensor == nullptr)
    {
        Serial.println("ERROR: Input tensor is NULL!");

        while (1)
        {
            delay(1000);
        }
    }


    if (output_tensor == nullptr)
    {
        Serial.println("ERROR: Output tensor is NULL!");

        while (1)
        {
            delay(1000);
        }
    }


    // ========================================================
    // INPUT INFORMATION
    // ========================================================

    Serial.println();
    Serial.println("========================================");
    Serial.println(" INPUT TENSOR");
    Serial.println("========================================");

    Serial.print("Type: ");
    Serial.println(input_tensor->type);

    Serial.print("Bytes: ");
    Serial.println(input_tensor->bytes);

    Serial.print("Dimensions: ");

    for (int i = 0;
         i < input_tensor->dims->size;
         i++)
    {
        Serial.print(input_tensor->dims->data[i]);

        if (i < input_tensor->dims->size - 1)
        {
            Serial.print(" x ");
        }
    }

    Serial.println();

    Serial.print("Scale: ");
    Serial.println(
        input_tensor->params.scale,
        10
    );

    Serial.print("Zero Point: ");
    Serial.println(
        input_tensor->params.zero_point
    );


    // ========================================================
    // OUTPUT INFORMATION
    // ========================================================

    Serial.println();
    Serial.println("========================================");
    Serial.println(" OUTPUT TENSOR");
    Serial.println("========================================");

    Serial.print("Type: ");
    Serial.println(output_tensor->type);

    Serial.print("Bytes: ");
    Serial.println(output_tensor->bytes);

    Serial.print("Dimensions: ");

    // IMPORTANT:
    // Correct:
    // output_tensor->dims->size
    //
    // NOT:
    // output_tensor->dims->dims->size

    for (int i = 0;
         i < output_tensor->dims->size;
         i++)
    {
        Serial.print(output_tensor->dims->data[i]);

        if (i < output_tensor->dims->size - 1)
        {
            Serial.print(" x ");
        }
    }

    Serial.println();

    Serial.print("Scale: ");
    Serial.println(
        output_tensor->params.scale,
        10
    );

    Serial.print("Zero Point: ");
    Serial.println(
        output_tensor->params.zero_point
    );


    // ========================================================
    // TEST INPUT
    // ========================================================

    Serial.println();
    Serial.println("========================================");
    Serial.println(" TEST INFERENCE");
    Serial.println("========================================");

    if (input_tensor->type != kTfLiteInt8)
    {
        Serial.println("ERROR: Model input is not INT8!");

        while (1)
        {
            delay(1000);
        }
    }


    // --------------------------------------------------------
    // Fill input with zero-value quantization
    //
    // This is ONLY a model execution test.
    // It is NOT a real fatigue prediction.
    // --------------------------------------------------------

    int8_t input_zero_point =
        (int8_t)input_tensor->params.zero_point;

    for (int i = 0;
         i < input_tensor->bytes;
         i++)
    {
        input_tensor->data.int8[i] =
            input_zero_point;
    }


    // --------------------------------------------------------
    // RUN INFERENCE
    // --------------------------------------------------------

    unsigned long start_time = micros();

    status = interpreter->Invoke();

    unsigned long inference_time =
        micros() - start_time;


    if (status != kTfLiteOk)
    {
        Serial.println("ERROR: Inference FAILED!");

        while (1)
        {
            delay(1000);
        }
    }


    Serial.println("Inference SUCCESS");

    Serial.print("Inference time: ");
    Serial.print(inference_time / 1000.0);
    Serial.println(" ms");


    // ========================================================
    // OUTPUT
    // ========================================================

    Serial.println();
    Serial.println("========================================");
    Serial.println(" MODEL OUTPUT");
    Serial.println("========================================");


    if (output_tensor->type == kTfLiteInt8)
    {
        for (int i = 0;
             i < output_tensor->bytes;
             i++)
        {
            int8_t q =
                output_tensor->data.int8[i];

            float value =
                (q - output_tensor->params.zero_point)
                * output_tensor->params.scale;

            Serial.print("Output[");
            Serial.print(i);
            Serial.print("] = ");

            Serial.print(value, 6);

            Serial.print("   INT8 = ");

            Serial.println(q);
        }
    }
    else
    {
        Serial.println("Output is not INT8.");
    }


    Serial.println();
    Serial.println("========================================");
    Serial.println(" MODEL TEST COMPLETE");
    Serial.println("========================================");

    Serial.println();
    Serial.println("Expected model input:");
    Serial.println("20 time steps x 10 features");

    Serial.println();
    Serial.println("Next stage:");
    Serial.println("sEMG sensor -> feature extraction");
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
    delay(5000);

    Serial.println(
        "Model loaded. Waiting for real sEMG features..."
    );
}