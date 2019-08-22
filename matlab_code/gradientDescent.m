function [params, cost] = gradientDescent(costFunction, params_init, iterations, alpha)
    
    params = params_init;
    for i=1:iterations
        [cost, grad] = costFunction(params);
        params = params - alpha * grad;
        fprintf('Iteration: %4i | Cost: %4.6f\r', i, cost);
    end
    fprintf('\n');
end