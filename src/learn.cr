require "./learn/*"

module Learn
  class NeuralNetwork
    def initialize(options : Hash)
      @learning_rate = options["learning_rate"] || 0.3
      @momentum      = options["momentum"] || 0.1
      @hidden_sizes  = options["hidden_layers"]

      @binary_thresh = options["binary_thresh"] || 0.5
    end

    def init(@sizes)
      @output_layer = @sizes[@sizes.size - 1]

      # network states
      @biases  = [] of Float32
      @weights = [] of Float32
      @outputs = [] of Float32

      # training states
      @deltas  = [] of Float32
      @changes = [] of Float32
      @errors  = [] of Float32

      layer = 0
      @sizes.each do |size|
        @deltas[layer]  = util_zeros size
        @errors[layer]  = util_zeros size
        @outputs[layer] = util_zeros size

        if layer > 0
          @biases[layer]  = util_randos
          @weights[layer] = Array.new(size, Float32)
          @changes[layer] = Array.new(size, Float32)

          (0 .. size).each do |node|
            prev_size = @sizes[layer - 1]

            @weights[layer][node] = util_randos(prev_size)
            @changes[layer][node] = util_zeros prev_size
          end
        end

        layer += 1
      end
    end
  end
end
