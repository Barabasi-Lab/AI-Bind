function [t01, t10, k01cal, k10cal,summat01, summat10]=multidegree_entropy_pos_neg(k01, k10)
precision=10^(-5);
loops=10000;
n=length(k01);



t01=rand(n,1);
t10=rand(n,1);




oldt01=zeros(n,1);
oldt10=zeros(n,1);


for kk=1:loops
    
    T01=t01*t01';
    T10=t10*t10';
    
    
    %partition function
    Z=1+ T01 + T10;
    
    
    %p01
    
    summat=(ones(n,1)*t01')./(Z+(Z==0));
    summat=summat-diag(diag(summat));
    summat=sum(summat,2);
    t01=k01./(summat+(summat==0)); 
    T01=t01*t01';
    Z=1+ T01 + T10;
    
    %p10
    summat=(ones(n,1)*t10')./(Z+(Z==0));
    summat=summat-diag(diag(summat));
    summat=sum(summat,2);
    t10=k10./(summat+(summat==0));   
%    T10=t10*t10';
%    Z=1+ T01 + T10;
    
    

    prec2= max(abs((t01>0).*(1-t01./(oldt01+(oldt01==0)))));
    %display(prec2);
    prec3=max(abs((t10>0).*(1-t10./(oldt10+(oldt10==0)))));
    %display(prec3);

  
    
    if (max(abs((t01>0).*(1-t01./(oldt01+(oldt01==0)))))<precision ...
           && max(abs((t10>0).*(1-t10./(oldt10+(oldt10==0)))))<precision)
           break
    end
   
    
    oldt01=t01;
    oldt10=t10;
  
end

    %disp(kk);
    T01=t01*t01';
    T10=t10*t10';

    Z=1+ T01 + T10;
    
    
    summat01=T01./(Z+(Z==0));
    summat01=summat01-diag(diag(summat01));
    k01cal=sum(summat01,2);  
    
    summat10=T10./(Z+(Z==0));
    summat10=summat10-diag(diag(summat10));
    k10cal=sum(summat10,2);
    
