function [h, display_array] = displayLabels(labels_true, labels_predict, example_width, font_size)
    
    % Compute rows, cols
    num_examples = length(labels_predict);
    example_height = example_width;

    % Compute number of items to display
    display_rows = floor(sqrt(num_examples));
    display_cols = ceil(num_examples / display_rows);

    % Between images padding
    pad = 1;

    % Setup blank display
    display_array = zeros(pad + display_rows * (example_height + pad), ...
                           pad + display_cols * (example_width + pad));
    
    % prikazi oznake
    blank = ones(size(display_array));

    % grid domains
    wh = size(blank,1);
    step = wh / display_rows;
    coord = (step/2):step:wh;

    % label coordinates
    [xlbl, ylbl] = meshgrid(coord, coord);
    xlbl = xlbl';
    ylbl = ylbl';

    % create cell arrays of number labels
    lbl = strtrim(cellstr(num2str(labels_predict)));

    % narisi
    figure;
    h = imagesc(blank, [0, 1]);
    colormap gray;
    axis image;

    set(gca,'xtick',linspace(0,wh,display_rows+1));
    set(gca,'ytick',linspace(0,wh,display_cols+1));
    set(gca,'GridAlpha', 0.5);
    grid on;

    xlbl = xlbl(:);
    ylbl = ylbl(:);
    lbl = lbl(:);
    for i = 1:length(lbl)
        if (round(labels_predict(i)) == round(labels_true(i)))
            text(xlbl(i), ylbl(i), lbl(i), 'color', 'k', ...
            'HorizontalAlignment','center','VerticalAlignment','middle', ...
            'FontSize', font_size, 'FontWeight', 'bold');
        else
            text(xlbl(i), ylbl(i), lbl(i), 'color',[0.8350, 0.0780, 0.1840], ...
            'HorizontalAlignment','center','VerticalAlignment','middle', ...
            'FontSize', font_size, 'FontWeight', 'bold');
        end
    	
    end
    
%     text(xlbl(:), ylbl(:), lbl(:), 'color','k', ...
%         'HorizontalAlignment','center','VerticalAlignment','middle', ...
%         'FontSize', font_size, 'FontWeight', 'bold');
end