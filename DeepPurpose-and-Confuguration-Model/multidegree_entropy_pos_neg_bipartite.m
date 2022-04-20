function [t01, t10, m01, m10,  k01cal, k10cal, l01cal, l10cal, summat01, summat10]=multidegree_entropy_pos_neg_bipartite(k01, k10, l01, l10)
precision=10^(-5);
loops=10000;
nr=length(k01);
disp(nr)
nc=length(l01);
disp(nc)


t01=rand(nr,1);
t10=rand(nr,1);

m01=rand(1,nc);
m10=rand(1,nc);

oldt01=zeros(nr,1);
oldt10=zeros(nr,1);

oldm01=zeros(1,nc);
oldm10=zeros(1,nc);


for kk=1:loops
    
    T01=t01*m01;
    T10=t10*m10;
    
    
    %partition function
    Z=1+ T01 + T10;
    
    
    %p01
    
    summat=(ones(nr,1)*m01)./(Z+(Z==0));
    summat=sum(summat,2);
    t01=k01./(summat+(summat==0)); 
    T01=t01*m01;
    Z=1+ T01 + T10;
    
    summat=(t01*ones(1,nc))./(Z+(Z==0));
    summat=sum(summat);
    m01=l01./(summat+(summat==0)); 
    T01=t01*m01;
    Z=1+ T01 + T10;    
    
    
    
    %p10
    summat=(ones(nr,1)*m10)./(Z+(Z==0));
    summat=sum(summat,2);
    t10=k10./(summat+(summat==0));   
    T10=t10*m10;
    Z=1+ T01 + T10;


    summat=(t10*ones(1,nc))./(Z+(Z==0));
    summat=sum(summat);
    m10=l10./(summat+(summat==0)); 
%   T10=t10*t10';
%   Z=1+ T01 + T10;
    
    

    %prec2= max(abs((t01>0).*(1-t01./(oldt01+(oldt01==0)))));
    %display(prec2)
    %prec3=max(abs((t10>0).*(1-t10./(oldt10+(oldt10==0)))));
    %display(prec3)

    %prec4= max(abs((m01>0).*(1-m01./(oldm01+(oldm01==0)))));
    %display(prec4)
    %prec5=max(abs((m10>0).*(1-m10./(oldm10+(oldm10==0)))));
    %display(prec5)
    

  
    
    if (max(abs((t01>0).*(1-t01./(oldt01+(oldt01==0)))))<precision ...
        && max(abs((t10>0).*(1-t10./(oldt10+(oldt10==0)))))<precision ...
        && max(abs((m01>0).*(1-m01./(oldm01+(oldm01==0)))))<precision ...
        && max(abs((m10>0).*(1-m10./(oldm10+(oldm10==0)))))<precision)
           break
    end
   
    
    oldt01=t01;
    oldt10=t10;
    oldm01=m01;
    oldm10=m10;    
  
end

    disp(kk)
    T01=t01*m01;
    T10=t10*m10;

    Z=1+ T01 + T10;
    
    
    summat01=T01./(Z+(Z==0));
    k01cal=sum(summat01,2);
    l01cal=sum(summat01); 
    
    summat10=T10./(Z+(Z==0));
    k10cal=sum(summat10,2);
    l10cal=sum(summat10);
    
