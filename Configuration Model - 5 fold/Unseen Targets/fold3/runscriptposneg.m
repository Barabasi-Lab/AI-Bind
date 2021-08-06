%% load('datadegree')
load('degreetrain01ligands_3.txt')
load('degreetrain10ligands_3.txt')
load('degreetrain01targets_3.txt')
load('degreetrain10targets_3.txt')
%%
[t01, t10, m01, m10,  k01cal, k10cal, l01cal, l10cal, summat01, summat10]=multidegree_entropy_pos_neg_bipartite(degreetrain01ligands_3, degreetrain10ligands_3, degreetrain01targets_3', degreetrain10targets_3');
%%
[S_bc,P, zr,zc]=configuration_bipartite(degreetrain10ligands_3,degreetrain10targets_3');
%% conditional probability
cond10=summat10./(summat10+summat01);
cond10(isnan(cond10))=0; 

%%
figure,
h1=histogram(log10(summat10./P), 'Normalization', 'PDF');
hold on
h2=histogram(log10(cond10./P), 'Normalization', 'PDF');
h1.EdgeColor='None';
h2.EdgeColor='None';
xlabel('log_{10}(probability ratio)')
ylabel('PDF')

%%
figure,
plot(degreetrain01ligands_3, k01cal, 'o')
hold on
plot(degreetrain10ligands_3, k10cal, 'o')

hold on
plot(degreetrain01targets_3, l01cal, 'o')
hold on
plot(degreetrain10targets_3, l10cal, 'o')

xlabel('real degree')
ylabel('predicted degree')


%%
figure,
h1=histogram(log10(summat01(:)), 'Normalization', 'PDF');
hold on
h2=histogram(log10(summat10(:)), 'Normalization', 'PDF');
h1.EdgeColor='None';
h2.EdgeColor='None';

xlabel('log_{10}(p)')
ylabel('PDF')

%% naive
Pnaive=(degreetrain10ligands_3./(degreetrain10ligands_3+degreetrain01ligands_3))*(degreetrain10targets_3'./(degreetrain10targets_3'+degreetrain01targets_3'));
Pnaive(isnan(Pnaive))=0; 

%%
corr(summat10(:), Pnaive(:), 'type', 'Spearman')
corr(cond10(:), Pnaive(:), 'type', 'Spearman')
corr(P(:), Pnaive(:), 'type', 'Spearman')

%%
writematrix(summat10,'P_3.csv')
writematrix(summat10,'summat10_3.csv')
writematrix(summat01,'summat01_3.csv')
