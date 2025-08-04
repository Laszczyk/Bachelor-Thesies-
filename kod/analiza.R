library(readxl)
install.packages('cluster')
library(ggdendro)
install.packages("dendextend")
library(dendextend)
install.packages("xlsx")
library(xlsx)
library(cluster)

library(ggplot2)
install.packages("factoextra")
install.packages("e1071")
install.packages("psych")
library(e1071)
library(psych)
library(cluster)
library(factoextra)
library(dplyr)
library(tidyr)

dane <- read.csv("C:/Users/kuba/OneDrive/Pulpit/praca licencjacka/dane3.csv", sep=";")
#dane <- dane[,-c(1,27)]
sum(sapply(dane, function(x) sum(is.na(x))))
dane <-na.omit(dane)
opisowe <- describe(dane)
opisowe$Variable <- rownames(opisowe)
write.csv(opisowe, file = "statystyki_opisowe.csv", row.names = FALSE)


skewness(dane[,21])
dane$HT_AvgGoalsHLastS <- log(dane[,21])
dane$AT_AvgGoalsALastS <- log(dane[,22])


#poniższy kod słuzy do zbudowanie ramek danych z widocznymi wartosciami obserwcaji odstajacych 

# Funkcja do wykrywania wartości odstających na podstawie IQR
detect_outliers_iqr <- function(series) {
  Q1 <- quantile(series, 0.25)
  Q3 <- quantile(series, 0.75)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  outliers <- (series < lower_bound) | (series > upper_bound)
  return(data.frame(Liczba_Obserwacji = sum(outliers), Dolna_Granica = lower_bound, Gorna_Granica = upper_bound))
}

# Funkcja do wykrywania wartości odstających na podstawie odchylenia standardowego
detect_outliers_std <- function(series, n_std = 3) {
  mean_val <- mean(series)
  std_val <- sd(series)
  lower_bound <- mean_val - n_std * std_val
  upper_bound <- mean_val + n_std * std_val
  outliers <- (series < lower_bound) | (series > upper_bound)
  return(data.frame(Liczba_Obserwacji = sum(outliers), Dolna_Granica = lower_bound, Gorna_Granica = upper_bound))
}

# Funkcje do wykrywania wartości odstających dla każdej kolumny
outliers_iqr <- lapply(dane, detect_outliers_iqr)
outliers_std <- lapply(dane, detect_outliers_std)

# Tworzenie ramki danych dla obu metod
outliers_iqr_df <- do.call(rbind, outliers_iqr)
outliers_iqr_df$Nazwa_Zmiennej <- rownames(outliers_iqr_df)

outliers_std_df <- do.call(rbind, outliers_std)
outliers_std_df$Nazwa_Zmiennej <- rownames(outliers_std_df)

rownames(outliers_iqr_df) <- NULL
rownames(outliers_std_df) <- NULL

# Wyświetlenie wynikowych ramek danych
print("Outliers IQR:")
print(outliers_iqr_df)
write.csv(outliers_iqr_df, file = "outliers_iqr_df.csv", row.names = FALSE)
print("\nOutliers Standard Deviation:")
print(outliers_std_df) 
write.csv(outliers_std_df, file = "outliers_std_df.csv", row.names = FALSE)


#ponizszy kod sluzy do usuwania obserwacji odstajacych ze zbioru danych

# Funkcja do wykrywania wartości odstających na podstawie IQR
detect_outliers_iqr <- function(series) {
  Q1 <- quantile(series, 0.25, na.rm = TRUE)
  Q3 <- quantile(series, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  outliers <- (series < lower_bound) | (series > upper_bound)
  return(outliers)
}

# Funkcja do wykrywania wartości odstających na podstawie odchylenia standardowego
detect_outliers_std <- function(series, n_std = 3) {
  mean_val <- mean(series, na.rm = TRUE)
  std_val <- sd(series, na.rm = TRUE)
  lower_bound <- mean_val - n_std * std_val
  upper_bound <- mean_val + n_std * std_val
  outliers <- (series < lower_bound) | (series > upper_bound)
  return(outliers)
}

remove_outliers <- function(data, method = "IQR") {
  if (method == "IQR") {
    outliers <- apply(data, 2, detect_outliers_iqr)
  } else if (method == "STD") {
    outliers <- apply(data, 2, detect_outliers_std)
  } else {
    stop("Nieznana metoda. Wprowadź 'IQR' lub 'STD'.")
  }
  
  # Usuwanie obserwacji odstających ze zbioru danych
  data_cleaned <- data
  data_cleaned[outliers] <- NA
  data_cleaned <- na.omit(data_cleaned)
  
  return(data_cleaned)
}

# Wykrywanie obserwacji odstających i usuwanie ich na podstawie wybranej metody
dane_po_usunieciu_iqr <- remove_outliers(dane, method = "IQR")
dane_po_usunieciu_std <- remove_outliers(dane, method = "STD")
dane <- dane_po_usunieciu_std
write.csv(dane, file = "dane_oczyszczone.csv", row.names = FALSE)
table(dane$seasonID)
table(dane_po_usunieciu_iqr$seasonID)
table(dane_po_usunieciu_std$seasonID)



dane_cor <- dane[,-c(1,27)]
# Obliczanie macierzy korelacji
cor_matrix <- cor(dane_cor)

cor_df <- reshape2::melt(cor_matrix)

# Wizualizacja korelacji za pomocą ggplot2
ggplot(cor_df, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  labs(title = "Mapa cieplna korelacji") +
  theme(panel.border = element_rect(color = 'black',fill = NA,size = 1),
      panel.background = element_rect(fill='gray95'),
      plot.background = element_rect(color = 'black', size = 1),
      axis.text.x = element_text(angle = 45, hjust = 1),
      axis.title.x = element_blank(),  
      axis.title.y = element_blank())




dane_cluster <- dane[,-c(1,6,7,17,18,27)]
cor_matrix <- cor(dane_cluster)


cor_cluster <- diana(as.dist(1 - abs(cor_matrix)))

# Konwersja wyników klastrowania do obiektu dendrogram
cor_dend <- as.dendrogram(cor_cluster)

#wyswitelenie clustrow 
# Apply hierarchical clustering with the optimal threshold
clusters <- cutree(cor_dend, h = 0.3)

# Assign cluster labels to variables
variable_cluster <- data.frame(Variable = colnames(cor_matrix), Cluster = clusters)

# Display the variables and their assigned clusters
variable_cluster <- variable_cluster[order(variable_cluster$Cluster), ]
print(variable_cluster)




#dendrogram 
ggdendrogram(cor_dend, rotate = TRUE, leaf_labels = TRUE, theme_dendro = FALSE) +
  #theme(axis.text.y = element_text(size = 10)) +
  labs(title = "Dendrogram")+
  geom_hline(yintercept = 0.3, linetype = "dashed", color = "red")+
  theme(panel.border = element_rect(color = 'black',fill = NA,size = 1),
        panel.background = element_rect(fill='gray95'),
        plot.background = element_rect(color = 'black', size = 1), 
        axis.title.x = element_blank(),  
        axis.title.y = element_blank())





cor_pairs <- as.data.frame(as.table(cor_matrix))
colnames(cor_pairs) <- c("Zmienna1", "Zmienna2", "Korelacja")
cor_pairs <- subset(cor_pairs, Korelacja >= 0.7 & Korelacja < 1)
cor_pairs <- cor_pairs %>%
  distinct(Korelacja, .keep_all = TRUE)
write.csv(cor_pairs, file = "pary_korelacji.csv", row.names = FALSE)








hc <- hclust(dist(cor_matrix), method = "ward.D2")

# List to store average correlation for each threshold
average_corr <- c()

# List of thresholds to test
thresholds <- seq(0.75, 1.0, by = 0.01)

# For each threshold
for (t in thresholds) {
  # Apply hierarchical clustering
  clusters <- cutree(hc, h = t)
  
  # Assign cluster labels to variables
  variable_cluster <- dane.frame(Variable = colnames(cor_matrix), Cluster = clusters)
  
  # Calculate the average correlation within each cluster
  cluster_corr <- tapply(1:nrow(variable_cluster), variable_cluster$Cluster, function(idx) {
    mean(cor_matrix[idx, idx])
  })
  
  # Calculate and store the average correlation across all clusters
  average_corr <- c(average_corr, mean(cluster_corr, na.rm = TRUE))
}

# Find the threshold with the highest average correlation
optimal_threshold <- thresholds[which.max(average_corr)]

# Apply hierarchical clustering with the optimal threshold
clusters <- cutree(hc, h = optimal_threshold)

# Assign cluster labels to variables
variable_cluster <- dane.frame(Variable = colnames(cor_matrix), Cluster = clusters)

# Display the variables and their assigned clusters
variable_cluster <- variable_cluster[order(variable_cluster$Cluster), ]
print(variable_cluster)
write.csv(variable_cluster, file = "klastry.csv", row.names = FALSE)




ggdendrogram(hc, rotate = TRUE, leaf_labels = TRUE, theme_dendro = FALSE) +
  theme(axis.text.y = element_text(size = 10)) +
  labs(title = "Dendrogram") +
  geom_hline(yintercept = optimal_threshold, linetype = "dashed", color = "red") # Dodanie linii oznaczającej próg odcięcia












dane <- read.csv("C:/Users/kuba/OneDrive/Pulpit/praca licencjacka/dane3.csv", sep=";")

#analiza danych 
season_wins_losses <- dane %>% 
  group_by(seasonID, HomeWin) %>%
  summarise(count = n(), .groups = 'drop')

season_wins_losses$seasonID <- factor(season_wins_losses$seasonID, levels = rev(unique(season_wins_losses$seasonID)))
# Generuj wykres
ggplot(season_wins_losses, aes(fill=as.factor(HomeWin), y=count, x=as.factor(seasonID))) + 
  geom_bar(position="dodge", stat="identity") +
  scale_fill_manual(values=c("#99ccff", "#ff9999"), 
                    labels = c("Brak wygranej", "Wygrana"), 
                    name = "Wygrana na własnym boisku") +
  labs(x="Season ID", y="Liczba meczów", title="Rozkład zmiennej wynikowej na przestrzeni ostatnich 9 sezonów")+
  theme(panel.border = element_rect(color = 'black',fill = NA,size = 1),
        panel.background = element_rect(fill='gray95'),
        plot.background = element_rect(color = 'black', size = 1)
  )




table(dane$HomeWin)

ggplot(dane, aes(x=factor(HomeWin))) +
  geom_bar(aes(fill=factor(HomeWin)), width = 0.5) +
  scale_fill_manual(values=c("#99ccff", "#ff9999"), 
                    labels = c("Brak wygranej", "Wygrana"), 
                    name = "Wygrana na własnym boisku") +
  labs(x="Wygrana na własnym boisku", y="Liczba meczów", title="Rozkład zmiennej wynikowej w całym zestawie danych")+
  theme(panel.border = element_rect(color = 'black',fill = NA,size = 1),
      panel.background = element_rect(fill='gray95'),
      plot.background = element_rect(color = 'black', size = 2)
)


dane_top3 <- read_excel("dane/wartosc/wartosctop3.xlsx")


dane_top3_long <- dane_top3 %>%
  pivot_longer(cols = -seasonID, names_to = "Team", values_to = "Value")

# Convert 'seasonID' column to integer type
dane_top3_long$seasonID <- as.integer(dane_top3_long$seasonID)

# Filter the dane to include only teams 1, 2, 3 and average
filtered_dane <- dane_top3_long %>% filter(Team %in% c('1', '2', '3', 'srednia'))

# Order the levels of 'seasonID' in increasing order
filtered_dane$seasonID <- factor(filtered_dane$seasonID, levels = unique(filtered_dane$seasonID))

# Define custom labels for the legend
custom_labels <- c("1" = "Pierwsza drużyna", "2" = "Druga drużyna", "3" = "Trzecia drużyna", "srednia" = "Średnia ligowa")

# Grouped bar plot for filtered dane
ggplot(filtered_dane, aes(fill=Team, y=Value, x=seasonID)) + 
  geom_bar(position="dodge", stat="identity") +
  scale_fill_manual(values=c("#0066cc", "#3399ff","#99ccff", "#66ccff"), labels = custom_labels) +
  labs(x="ID sezonu", y="Wartość drużyn", title="Wartość drużyn top3 i średniej ligowej") +
  theme_minimal() +
  theme(legend.title = element_blank(), legend.position = "right",panel.border = element_rect(color = 'black',fill = NA,size = 1),
        panel.background = element_rect(fill='gray95'),
        plot.background = element_rect(color = 'black', size = 2))





numeric_cols <- colnames(dane[sapply(dane, is.numeric)])

# Remove 'seasonID' and 'HomeWin' from the list as they are not continuous variables
numeric_cols <- numeric_cols[!numeric_cols %in% c('seasonID', 'HomeWin')]

# Convert 'HomeWin' to factor for proper plotting
dane$HomeWin <- as.factor(dane$HomeWin)

# Plot boxplots for each numeric column
for (col in numeric_cols) {
  p <- ggplot(dane, aes(x = HomeWin, y = !!sym(col))) + 
    geom_boxplot(aes(fill = HomeWin)) + 
    scale_fill_manual(values=c("#99ccff", "#ff9999")) +
    labs(x = "HomeWin", y = col, title = paste("Wykres pudełkowy", col, "względem zmiennej HomeWin")) +
    theme(legend.title = element_blank(), 
          legend.position = "right",
          panel.border = element_rect(color = 'black',fill = NA,size = 1),
          panel.background = element_rect(fill='gray95'),
          plot.background = element_rect(color = 'black', size = 2))
  #filename <- paste("boxplot_", col, ".png", sep = "")
 # ggsave(filename, p, width = 8, height = 6, dpi = 300)
  print(p)
}


table(dane$HomeWin,dane$seasonID)





