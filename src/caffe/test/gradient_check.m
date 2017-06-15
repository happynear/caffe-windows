num = 1;
channel = 5;
height = 8;
width = 8;
epsilon = 1e-4;
er = 1e-5;
func = @gram_layer;

in = randn(num,channel,height,width);
[compute_output, compute_grad] = func(in);

for n=1:num
    for i = 1:size(compute_output,2)
        for j=i:size(compute_output,3)
%             if i~=j
%                 continue;
%             end;
            dzdy = zeros(num,channel,channel);
            dzdy(n,i,j) = 1;
%             dzdy(n,j,i) = dzdy(n,j,i) + 1;
            [compute_output, compute_grad] = func(in,dzdy);
            for c=1:channel
                for h = 1:height
                    for w = 1:width
                        positive_input = in;
                        positive_input(n,c,h,w) = positive_input(n,c,h,w) + epsilon;
                        [positive_output, positive_grad] = func(positive_input);

                        negative_input = in;
                        negative_input(n,c,h,w) = negative_input(n,c,h,w) - epsilon;
                        [negative_output, negative_grad] = func(negative_input);

                        dW = (positive_output(n,i,j) - negative_output(n,i,j)) / (2 * epsilon);
                        drW = compute_grad(n,c,h,w);
                        e = abs(dW - drW);
                        fprintf('(%d,%d,%d) (%d,%d,%d,%d) (%f,%f)\n',n,i,j,n,c,h,w,dW,drW);
                        assert(e < er, 'numerical gradient checking failed');
                    end;
                end;
            end;
            
        end;
    end;
end;