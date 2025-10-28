library(tidyverse)
library(stringr)

# load data
frame_index <- 40
data <- read_csv(paste0("data/video/free_association/", frame_index, ".csv"))

# split free-association content into words
data_split <- data %>%
  mutate(content = str_trim(content)) %>%
  separate_wider_delim(content, delim = ",", names = paste0("word_", 1:7), too_few = "align_start") %>%
  mutate(across(starts_with("word_"), str_trim))

# convert to long format
data_long <- data_split %>%
  pivot_longer(
    cols = starts_with("word_"),
    names_to = "word_id",
    values_to = "word"
  ) %>%
  filter(!is.na(word)) %>%
  mutate(word_id = str_extract(word_id, "\\d+"))

# summarize word frequencies 
by_position <- data_long %>%
  group_by(word_id, word) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(word_id) %>%
  mutate(proportion = n/sum(n)) %>%
  arrange(word_id, desc(proportion)) %>%
  group_by(word_id) %>%
  slice_head(n = 10)

overall_freq <- data_long %>%
  group_by(word) %>%
  summarise(n = n()) %>%
  mutate(proportion = n/sum(n)) %>%
  arrange(desc(proportion)) %>%
  slice_head(n = 10)

position_plots <- ggplot(by_position, aes(x = reorder(word, proportion), y = proportion)) +
  geom_bar(stat = "identity") +
  facet_wrap(~word_id, scales = "free_y") +
  coord_flip() +
  theme_minimal() +
  labs(x = "Word", y = "Proportion")

overall_plot <- ggplot(overall_freq, aes(x = reorder(word, proportion), y = proportion)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(x = "Word", y = "Proportion", title = "Overall Word Frequencies")

print(position_plots)
print(overall_plot)
