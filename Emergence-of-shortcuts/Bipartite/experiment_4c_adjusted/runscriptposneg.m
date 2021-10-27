%% load('datadegree')
load('degreetrain01ligands.txt')
load('degreetrain10ligands.txt')
load('degreetrain01targets.txt')
load('degreetrain10targets.txt')
%%
[t01, t10, m01, m10,  k01cal, k10cal, l01cal, l10cal, summat01, summat10]=multidegree_entropy_pos_neg_bipartite(degreetrain01ligands, degreetrain10ligands, degreetrain01targets', degreetrain10targets');
%%
[S_bc,P, zr,zc]=configuration_bipartite(degreetrain10ligands,degreetrain10targets');
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
plot(degreetrain01ligands, k01cal, 'o')
hold on
plot(degreetrain10ligands, k10cal, 'o')

hold on
plot(degreetrain01targets, l01cal, 'o')
hold on
plot(degreetrain10targets, l10cal, 'o')

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

%%
Pnaive=(degreetrain10ligands./(degreetrain10ligands+degreetrain01ligands))*(degreetrain10targets'./(degreetrain10targets'+degreetrain01targets'));

%% naive
Pnaive=(degreetrain10ligands./(degreetrain10ligands+degreetrain01ligands))*(degreetrain10targets'./(degreetrain10targets'+degreetrain01targets'));
Pnaive(isnan(Pnaive))=0; 

%%
corr(summat10(:), Pnaive(:), 'type', 'Spearman')
corr(cond10(:), Pnaive(:), 'type', 'Spearman')
corr(P(:), Pnaive(:), 'type', 'Spearman')

%%
writematrix(summat10,'C:\Users\Ayan\Documents\GitHub\AI-Bind\Emergence-of-shortcuts\experiment_4c_adjusted\P.csv')
writematrix(summat10,'C:\Users\Ayan\Documents\GitHub\AI-Bind\Emergence-of-shortcuts\experiment_4c_adjusted\summat10.csv')
writematrix(summat01,'C:\Users\Ayan\Documents\GitHub\AI-Bind\Emergence-of-shortcuts\experiment_4c_adjusted\summat01.csv')
