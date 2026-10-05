package fatigue;

import ai.djl.Model;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.convolutional.Conv1d;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.Activation;
import ai.djl.ndarray.types.Shape;

public class FatigueTCN {

    public static final int TIME_STEPS = 20;
    public static final int FEATURES = 10;

    public static Block createModel() {

        SequentialBlock block = new SequentialBlock();

        // First temporal convolution
        block.add(
                Conv1d.builder()
                        .setFilters(32)
                        .setKernelShape(new Shape(3))
                        .optPadding(new Shape(1))
                        .build()
        );

        block.add(Activation::relu);

        block.add(
                Dropout.builder()
                        .optRate(0.2f)
                        .build()
        );

        // Second temporal convolution
        block.add(
                Conv1d.builder()
                        .setFilters(32)
                        .setKernelShape(new Shape(3))
                        .optPadding(new Shape(1))
                        .build()
        );

        block.add(Activation::relu);

        block.add(
                Dropout.builder()
                        .optRate(0.2f)
                        .build()
        );

        // Third temporal convolution
        block.add(
                Conv1d.builder()
                        .setFilters(64)
                        .setKernelShape(new Shape(3))
                        .optPadding(new Shape(1))
                        .build()
        );

        block.add(Activation::relu);

        // Classification layers
        block.add(
                Linear.builder()
                        .setUnits(32)
                        .build()
        );

        block.add(Activation::relu);

        block.add(
                Linear.builder()
                        .setUnits(2)
                        .build()
        );

        return block;
    }

    public static Model createModelInstance() {

        Model model = Model.newInstance("HamstringFatigueTCN");

        model.setBlock(createModel());

        return model;
    }
}