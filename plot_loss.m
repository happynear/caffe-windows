filename = 'log/INFO2015-04-29T22-20-15.txt';
fid=fopen(filename,'r');
regpat = 'Iteration [0-9]+, loss = [0-9\.]+';
iter = zeros(100000,1);
loss = zeros(100000,1);
p = 1;
while ~feof(fid)
    newline=fgetl(fid);
    o3=regexpi(newline,regpat,'match');
    if ~isempty(o3)
        iterloss = sscanf(o3{1},'Iteration %d, loss = %f');
        iter(p) = iterloss(1);
        loss(p) = iterloss(2);
        p=p+1;
    end;
end;
fclose(fid);
iter = iter(1:p-1);
loss = loss(1:p-1);
plot(iter,loss);