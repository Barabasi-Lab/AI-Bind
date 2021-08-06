%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INPUT
% pp unweighted undirected bipartite adjacency matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%OUTPUT
% S_bc bipartite configuration canonical entropy
% P link probability matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%function [S_bc,P, zr,zc]=configuration_bipartite(pp)
function [S_bc,P, zr,zc]=configuration_bipartite(kr,kc)
precision=10^(-5);
loops=10000;

%[nr, nc]=size(pp);
%kr=sum(pp,2);
%kc=sum(pp);

nr=length(kr);
nc=length(kc);

% zr=rand(nr,1);
% zc=rand(1, nc);
L=sum(kr);
zr=kr/sqrt(L);
zc=kc/sqrt(L);
oldzr=zeros(nr,1);
oldzc=zeros(1,nc);


for kk=1:loops
            
    
    
    
            U=ones(nr,1)*zc;
            D=ones(nr,nc) + zr*zc;  
            zr=kr ./ sum(U./D,2);
            %zr=max(zr,10^(-15));
            
            U=zr*ones(1,nc);
            D=ones(nr,nc) + zr*zc;  
            zc=kc ./ sum(U./D);
            %zc=max(zc,10^(-15));            
            
            
        
        if (max(abs((zr>0).*(1-zr./(oldzr+(oldzr==0)))))<precision)&&(max(abs((zc>0).*(1-zc./(oldzc+(oldzc==0)))))<precision)
            break
        end
        oldzr=zr;
        oldzc=zc;
	end

    
%Compute link probability
P=(zr*zc)./(ones(nr,nc)+zr*zc);

%Compute Shannon entropy
P1=P.*log(P+(P==0));
P2=(ones(nr,nc)-P).*log(ones(nr,nc)-P +((ones(nr,nc)-P)==0));
S_bc=-sum(sum(P1+P2));
disp(kk)

return