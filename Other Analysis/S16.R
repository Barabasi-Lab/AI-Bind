require(data.table)
require(tidyr)
require(magrittr)
require(dplyr)
require(igraph)
`%ni%` <- Negate(`%in%`)

alltargets = fread("data/Deisy_VecNet_Unseen_Edges_Test_v09_All_Predictions.csv")
dict = alltargets %>%
  dplyr::select(`Gene Name`, target_aa_code) %>% 
  base::unique()

targets = fread("data/test_predictions_5_fold_gene_chem_names.csv")

targets %<>% filter(protein_id != "Not found")

Unseen_Targets = targets %>% 
  mutate(`Correct Prediction` = ifelse(Y == binary_Y, "YES", "NO")) %>%
  mutate(`Error` = ifelse(Y == 1 & binary_Y == 1, "True Positive", NA)) %>%
  mutate(`Error` = ifelse(Y == 0 & binary_Y == 1, "False Positive", Error)) %>%
  mutate(`Error` = ifelse(Y == 1 & binary_Y == 0, "False Negative", Error)) %>%
  mutate(`Error` = ifelse(Y == 0 & binary_Y == 0, "True Negative", Error)) %>%
  # filter(`Gene Name` != "")%>% 
  group_by(protein_id, target_aa_code, `Correct Prediction`) %>% 
  summarise(., freq = n())%>% 
  arrange(desc(freq)) %>% 
  pivot_wider(names_from = `Correct Prediction`, values_from = freq, values_fill = 0)

targets$target_aa_code %>% unique() %>% length()

sums = colSums(Unseen_Targets[,-c(1,2)])

p = list()
for (i in 1: nrow(Unseen_Targets)){
  y = matrix(c(Unseen_Targets$YES[i], Unseen_Targets$NO[i],
               sums[1] - Unseen_Targets$YES[i], sums[2]- Unseen_Targets$NO[i]), ncol = 2, byrow = TRUE)
  p[[i]] = fisher.test(y, alternative = "less")$p
}
p %<>% unlist()
Unseen_Targets$p = p
Unseen_Targets$p_adj = p.adjust(Unseen_Targets$p, method = "fdr")

Unseen_Targets %>% filter(p_adj < 0.05) %>%
  nrow()

require(ggplot2)
ggplot(Unseen_Targets) +
  aes(x = YES, y = NO, colour = (p_adj), size = -log10(p_adj)) +
  geom_point() +
  scale_size(range = c(0,4))+
  scale_color_distiller(palette = "OrRd", direction = -1) +
  labs(color = "p adj", size = "")+
  guides(size = F)+
  xlim(c(0,350))+
  ylim(c(0,350))+
  theme_minimal()+
  theme(legend.position = "bottom")


Genes = Unseen_Targets %>% 
  ungroup() %>% 
  filter(p_adj< 0.05) %>% 
  dplyr::select(protein_id, target_aa_code) %>%
  mutate(tick = 1:n())

data = Genes
data$size = stringr::str_count(data$target_aa_code)

data = subset(data, data$size > 25)
u = vector()
for( i in c(1:nrow(data))){
  u[i]= paste0(">", data$protein_id[i], 
               "\n", data$target_aa_code[i])
}

dic = targets %>% 
  select(gene, protein_id) %>%
  unique 

write.table(u, 
            file = "data/SeqSep21.fa", 
            row.names = F, quote = F, col.names = F)


# system ("./muscle3.8.31_i86darwin64 -in SeqSep21.fa -out SeqSep21.afa -seqtype protein")
# system ("./muscle3.8.31_i86darwin64 -in SeqSep21.afa -out RefinedSep21.afa -seqtype protein -refine")
# system('./muscle3.8.31_i86darwin64 -maketree -in RefinedSep21.afa -out IncorrectPredictions.phy  -cluster neighborjoining')
library("treeio")
library("ggtree")

tree = treeio::read.tree("./data/IncorrectPredictions.phy")

# ggplot(tree, aes(x, y)) + 
#   geom_tree() +
#   theme_tree()


df <- data.frame(protein_id = tree$tip.label)
rownames(df) <- tree$tip.label


humangenes = fread('./data/interactome_2019_merged_protAnnots.csv')
dic$Specie = ifelse(dic$gene %in% humangenes$Symbol, "Human", "Other")
dic$Specie = ifelse(dic$gene == "", "No Gene Name", dic$Specie)

dic_sp = dic %>%
  select(protein_id, Specie) %>%
  unique() %>%
  group_by(protein_id) %>% 
  mutate(n = n()) %>%
  mutate(Specie = ifelse(n > 1, "Human", Specie)) %>%
  select(Specie, protein_id) %>%
  unique()

df = df %>% left_join(., dic_sp)
row.names(df) = df$protein_id

circ <- ggtree(tree, layout = "circular")
df %<>% dplyr::select(Specie)

p1 <- gheatmap(circ, 
               df, 
               offset=.8, width=.2,
               colnames_angle=90, 
               colnames_offset_y = .25) +
  scale_fill_viridis_d(option="D", 
                       name="Specie")

p1


Cairo::CairoPDF("./figs/IncorrectPredictions.pdf", 
                width = 15, 
                height = 10)
p <- ggtree(tree, layout = "circular", branch.length='none') + 
  geom_tiplab(size=2, align=TRUE, linesize=.5) + 
  theme_tree2()
gheatmap(p, df, offset=0, 
         width=0.1, 
         colnames=FALSE, 
         legend_title="Specie") +
  scale_fill_viridis_d(option="D", 
                       name="Specie")+
  scale_x_ggtree() + 
  scale_y_continuous(expand=c(0, 0.3)) +
  theme(legend.position = "bottom")
dev.off()

######## 
######## FALSE POSITIVES
######## 

Error = targets %>% 
  mutate(`Correct Prediction` = ifelse(Y == binary_Y, "YES", "NO")) %>%
  mutate(`Error` = ifelse(Y == 1 & binary_Y == 1, "True Positive", NA)) %>%
  mutate(`Error` = ifelse(Y == 0 & binary_Y == 1, "False Positive", Error)) %>%
  mutate(`Error` = ifelse(Y == 1 & binary_Y == 0, "False Negative", Error)) %>%
  mutate(`Error` = ifelse(Y == 0 & binary_Y == 0, "True Negative", Error)) %>%
  # filter(`Gene Name` != "")%>% 
  group_by(protein_id, 
           target_aa_code, 
           `Error`) %>% 
  summarise(., freq = n())%>% 
  arrange(desc(freq)) %>% 
  pivot_wider(names_from = `Error`, 
              values_from = freq, 
              values_fill = 0)

targets$target_aa_code %>% unique() %>% length()

sums = colSums(Error[,-c(1,2)])

p = list()
for (i in 1: nrow(Error)){
  y = matrix(c(Error$`True Positive`[i], Error$`False Positive`[i],
               sums[1] - Error$`True Positive`[i], 
               sums[3] - Error$`False Positive`[i]), 
             ncol = 2, byrow = TRUE)
  p[[i]] = fisher.test(y, alternative = "less")$p
}
p %<>% unlist()
Error$p_TPFP = p
Error$p_adj_TPFP = p.adjust(Error$p_TPFP, method = "fdr")

p = list()
for (i in 1: nrow(Error)){
  y = matrix(c(Error$`True Negative`[i], Error$`False Negative`[i],
               sums[2] - Error$`True Negative`[i], 
               sums[4] - Error$`False Negative`[i]), 
             ncol = 2, byrow = TRUE)
  p[[i]] = fisher.test(y, alternative = "less")$p
}
p %<>% unlist()
Error$p_TNFN = p
Error$p_adj_TNFN = p.adjust(Error$p_TNFN, method = "fdr")


Error %>% filter(p_adj_TPFP < 0.05 | p_adj_TNFN < 0.05)
Error %>% filter(p_adj_TPFP < 0.05) %>%
  nrow()
Error %>% filter(p_adj_TNFN < 0.05) %>%
  nrow()

fwrite(Error, "./data/Error.csv")

Error %>% 
  mutate(class = ifelse(p_adj_TNFN < 0.05, "False Negative", "Correct")) %>%
  mutate(class = ifelse(p_adj_TPFP < 0.05, "False Positive", class)) %>%
  select(protein_id, target_aa_code, class) %>%
  fwrite("./data/Error_summary.csv")

Genes = Error %>% 
  ungroup() %>% 
  # filter(p_adj_TPFP < 0.05 | p_adj_TNFN < 0.05) %>% 
  dplyr::select(protein_id, target_aa_code) %>%
  mutate(tick = 1:n())

data = Genes
data$size = stringr::str_count(data$target_aa_code)

data = subset(data, data$size > 25)
u = vector()
for( i in c(1:nrow(data))){
  u[i]= paste0(">", data$protein_id[i], 
               "\n", data$target_aa_code[i])
}

dic = targets %>% 
  select(gene, protein_id) %>%
  unique 

write.table(u, 
            file = "./data/SeqSep21_all.fa", row.names = F, quote = F, col.names = F)

# system ("./muscle3.8.31_i86darwin64 -in SeqSep21_all.fa -out SeqSep21_all.afa -seqtype protein -maxiters 2")
# system("./muscle3.8.31_i86darwin64 -maketree -in SeqSep21_all.afa -out All_UPGMA.phy")

library("treeio")
library("ggtree")
require(ggplot2)

tree = treeio::read.tree("./data/All_UPGMA.phy")


Error = fread("./data/Error.csv")
df <- data.frame(protein_id = tree$tip.label)
df %<>% 
  dplyr::left_join(., Error) %>%
  mutate(class = ifelse(p_adj_TNFN < 0.05, "False Negative", "Correct")) %>%
  mutate(class = ifelse(p_adj_TPFP < 0.05, "False Positive", class)) %>%
  mutate(tick = 1) %>%
  group_by(protein_id) %>%
  mutate(n = cumsum(tick)) %>%
  mutate(n = ifelse(n ==1, "",n))%>%
  mutate(protein_id = paste(protein_id,n, sep = "_"))%>%
  select(protein_id , class) %>%
  as.data.frame()
row.names(df) <- df$protein_id



circ <- ggtree(tree)
df %<>% dplyr::select(class)

p1 <- gheatmap(circ, 
               df, 
               offset=.8, width=.2,
               colnames_angle=90, 
               colnames_offset_y = .25) +
  scale_fill_viridis_d(option="D", 
                       name="Error")
p1


Cairo::CairoPDF("./figs/Error.pdf", 
                width = 35, 
                height = 10)
p <- ggtree(tree, layout = "circular") + 
  geom_tiplab(size=2, align=TRUE, linesize=.5) + 
  theme_tree2()
gheatmap(p, df, offset=0, 
         width=0.1, 
         colnames=FALSE, 
         legend_title="Error") +
  scale_fill_manual(values =c("Correct" = "#C6C6C6", 
                              "False Negative" = "#119DA4",
                              "False Positive" = "#A4119D")) +
  scale_x_ggtree() + 
  scale_y_continuous(expand=c(0, 0.3)) +
  theme(legend.position = "bottom")
dev.off()

##################
##################
##################
Genes_wrong = Error %>% 
  ungroup() %>% 
  filter(p_adj_TPFP < 0.05 | p_adj_TPFP < 0.05 | p_adj_TPFP < 0.5 | p_adj_TNFN < 0.5  ) %>%
  dplyr::select(protein_id, target_aa_code) %>%
  mutate(tick = 1:n()) 

data = Genes_wrong
data$size = stringr::str_count(data$target_aa_code)
cut_l = quantile(data$size, 0.01)
cut_u = quantile(data$size, 0.99)
cut_l; cut_u

data = subset(data, data$size > cut_l) %>%
  filter(size < cut_u)
u = vector()
for( i in c(1:nrow(data))){
  u[i]= paste0(">", data$protein_id[i], 
               "\n", data$target_aa_code[i])
}

dic = targets %>% 
  select(gene, protein_id) %>%
  unique 

write.table(u, 
            file = "./data/SeqApr22_wrong.fa", row.names = F, quote = F, col.names = F)

# system ("./muscle3.8.31_i86darwin64 -in SeqApr22_wrong.fa -out SeqApr22_wrong.afa -seqtype protein -maxiters 2")
# system('./muscle3.8.31_i86darwin64 -maketree -in SeqApr22_wrong.afa -out WrongSeqApr22.phy  -cluster neighborjoining')
# system("./muscle3.8.31_i86darwin64 -maketree -in SeqApr22_wrong.afa -out WrongSeqApr22_UPGMA.phy")

library("treeio")
library("ggtree")

tree = treeio::read.tree("./data/WrongSeqApr22.phy")

ggplot(tree, aes(x, y)) + 
  geom_tree() +
  theme_tree()


df <- data.frame(protein_id = tree$tip.label)
df %<>% 
  dplyr::left_join(., Error) %>%
  mutate(class = ifelse(p_adj_TNFN < 0.05, "False Negative", "Correct")) %>%
  mutate(class = ifelse(p_adj_TPFP < 0.05, "False Positive", class)) %>%
  mutate(tick = 1) %>%
  group_by(protein_id) %>%
  mutate(n = cumsum(tick)) %>%
  mutate(n = ifelse(n ==1, "",n))%>%
  mutate(protein_id = paste(protein_id,n, sep = ""))%>%
  select(protein_id , class) %>%
  as.data.frame()
row.names(df) <- df$protein_id


circ <- ggtree(tree)
df %<>% dplyr::select(Bias = class)



Cairo::CairoPDF("./figs/S16_Error_only_circApr22_finetune.pdf", 
                width = 12, 
                height = 12)
p <- ggtree(tree, 
            layout = "fan", 
            size = 0.1, 
            ladderize = TRUE, open.angle = 0
)  %<+% df +
  geom_tiplab2(size=1.5, 
               align=TRUE, 
               linesize=0.01, 
  ) + 
  theme_tree2()
gheatmap(p, 
         df, 
         offset=0.5, 
         width=0.05, 
         colnames=FALSE, 
         legend_title="Bias", ) +
  # scale_x_ggtree() + 
  # scale_y_continuous(expand=c(0, 0.3)) +
  # 
  scale_fill_manual(values =c("Correct" = "#C6C6C6", 
                              "False Negative" = "#119DA4",
                              "False Positive" = "#A4119D")) +
  theme(legend.position = "bottom") 

# msaplot(p, "SeqSep21_wrong.afa")
dev.off()

############################
############################
############################


Unseen_Chemical = targets %>% 
  mutate(`Correct Prediction` = ifelse(Y == binary_Y, "YES", "NO")) %>%
  mutate(`Error` = ifelse(Y == 1 & binary_Y == 1, "True Positive", NA)) %>%
  mutate(`Error` = ifelse(Y == 0 & binary_Y == 1, "False Positive", Error)) %>%
  mutate(`Error` = ifelse(Y == 1 & binary_Y == 0, "False Negative", Error)) %>%
  mutate(`Error` = ifelse(Y == 0 & binary_Y == 0, "True Negative", Error)) %>%
  # filter(`Gene Name` != "")%>% 
  group_by(chem_name, InChiKey, `Correct Prediction`) %>% 
  summarise(., freq = n())%>% 
  arrange(desc(freq)) %>% 
  pivot_wider(names_from = `Correct Prediction`, values_from = freq, values_fill = 0)

Unseen_Chemical$InChiKey %>% unique() %>% length()

sums = colSums(Unseen_Chemical[,-c(1,2)])

p = list()
for (i in 1: nrow(Unseen_Chemical)){
  y = matrix(c(Unseen_Chemical$YES[i], Unseen_Chemical$NO[i],
               sums[1] - Unseen_Chemical$YES[i], sums[2]- Unseen_Chemical$NO[i]), ncol = 2, byrow = TRUE)
  p[[i]] = fisher.test(y, alternative = "less")$p
}
p %<>% unlist()
Unseen_Chemical$p = p
Unseen_Chemical$p_adj = p.adjust(Unseen_Chemical$p, method = "fdr")

Unseen_Chemical %>% filter(p_adj < 0.05) %>%
  nrow()
