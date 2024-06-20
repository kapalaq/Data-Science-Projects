library(readr)

df <- read_csv("./data/train.csv")
colSums(is.na(df) | df == "")
str(df)

# Nothing needed to be done
