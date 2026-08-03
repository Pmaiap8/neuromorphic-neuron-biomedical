`timescale 1ns / 1ns

module max_comparer #(
    parameter ITEM_CNT = 100,
    parameter DATA_WIDTH = 7, 
    parameter TAG_WIDTH = 7
)(
    input  logic [DATA_WIDTH-1 : 0] in_data[ITEM_CNT-1 : 0],
    output logic [TAG_WIDTH -1 : 0] out_tag
);
    
    logic [DATA_WIDTH-1:0] max_val;
    logic [TAG_WIDTH-1:0]  max_idx;

    always_comb begin
        // Inicializa assumindo que o primeiro elemento é o maior
        max_val = in_data[0];
        max_idx = '0;
        
        // Varre a matriz inteira em paralelo.
        // O Vivado vai sintetizar isso como uma árvore de comparadores otimizada.
        for (int i = 1; i < ITEM_CNT; i++) begin
            // A REGRA FAIL-SAFE: O sinal '>=' garante que, em caso de empate,
            // o índice maior (Neurônios da Doença, de 50 a 99) roube a liderança!
            // Agora, em caso de empate, o índice menor (Normal) mantém a liderança
            if (in_data[i] > max_val) begin 
                max_val = in_data[i];
                // Faz o cast explícito para evitar warnings de tamanho de bit
                max_idx = i[TAG_WIDTH-1:0]; 
            end
        end
    end
    
    assign out_tag = max_idx;
    
endmodule