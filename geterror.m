function [ error ] = geterror( prediction, labels )

fp = 0;
fn = 0;
m = size(prediction,1);

for i=1:m
   if (prediction(i)-labels(i) == 2)
       fp = fp+1;
   end
   if (prediction(i)-labels(i) == -2)
       fn = fn+1;
   end 
end

error = (5*fp + fn)/m; 

end

