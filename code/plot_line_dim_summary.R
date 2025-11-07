# objective ====
# Simplified code to only create line graphs for dimension distribution

# preparation ====
# packages 
library(tidyverse)
library(stringr)
library(grid)
library(gridExtra)
library(RColorBrewer)

# load data
# data parameters
frame_index <- 40 # which frame is annotated
n_word <- 10 # how many relevant words are allowed
n_simulation <- 50 # how many times the annotation is repeated

# visualization parameters
cut_word <- 10 # if integer, exhibit the first `cut_word` positions

frame_index_str <- paste0("frame", frame_index, "_word", n_word, "_sim", n_simulation)
data <- read_csv(paste0("/Users/rezek_zhu/multimodal_attention/data/video/gpt_free-association/processed_", frame_index_str, ".csv"))

# data processing ====
video_names <- data %>%
  distinct(video_name) %>%
  pull(video_name)

# Define consistent dimension order
dimension_order <- c("character", "location", "activity", "object", "relationship", "emotion_state", "other")
# Function to create dimension distribution line graph
create_dimension_line_graph <- function(data, video_name = NULL, cut_word = 10, dimension_order) {
  # Create a long format dataset
  data_long <- data %>%
    select(video_name, simulation_id, starts_with("category_")) %>% 
    pivot_longer(
      cols = starts_with("category_"),
      names_to = "category_id",
      values_to = "category"
    ) %>% 
    mutate(category_id = as.numeric(str_extract(category_id, "\\d+"))) %>%
    filter(category_id <= cut_word) %>%
    filter(category != "not_found")  # Remove not_found categories
  
  # Filter by video_name if specified
  if (!is.null(video_name)) {
    data_long <- data_long %>% filter(video_name == !!video_name)
    title_prefix <- paste0(video_name, " - ")
  } else {
    title_prefix <- "Dimension Distribution Across Word Positions (All Videos)"
  }
  
  # Calculate dimension counts by position
  dimension_counts_by_position <- data_long %>%
    filter(category %in% dimension_order) %>%
    group_by(category_id, category) %>%
    summarise(count = n(), .groups = "drop") %>%
    complete(category_id, category = dimension_order, fill = list(count = 0)) %>%
    mutate(category = factor(category, levels = dimension_order))
  
  # Create line graph
  line_graph <- ggplot(dimension_counts_by_position, 
                      aes(x = factor(category_id), y = count, group = category, color = category)) +
    geom_line(size = 1) +
    geom_point(size = 2) +
    scale_color_brewer(palette = "Set2") +
    theme_classic() +
    labs(
      title = paste0(title_prefix),
      x = "Word Position",
      y = "Count",
      color = "Dimension"
    ) +
    theme(
      legend.position = "right",
      panel.grid.minor = element_blank(),
      axis.text.x = element_text(angle = 0)
    )
  
  return(line_graph)
}

# Create the total dimension distribution graph
total_line_graph <- create_dimension_line_graph(data, NULL, cut_word, dimension_order)
print(total_line_graph)

# Create individual graphs for selected videos
# You can choose specific videos or loop through all
selected_videos <- video_names[c(1,10,116)]  # Example: first 3 videos
for (video in selected_videos) {
  video_line_graph <- create_dimension_line_graph(data, video, cut_word, dimension_order)
  print(video_line_graph)
}


# Function to calculate how many videos have each dimension appearing more than a threshold count at each position
calculate_dimension_prevalence <- function(data, dimension_order, threshold = 0) {
  # Convert data to long format if not already
  if (!"category_id" %in% colnames(data)) {
    data_long <- data %>%
      pivot_longer(
        cols = starts_with("category_"),
        names_to = "category_id",
        values_to = "category"
      ) %>%
      mutate(category_id = as.numeric(str_extract(category_id, "\\d+")))
  } else {
    data_long <- data
  }
  
  # Count videos where each dimension appears at each position
  dimension_prevalence <- data_long %>%
    filter(category %in% dimension_order) %>%
    group_by(video_name, category_id, category) %>%
    summarise(video_count = n(), .groups = "drop") %>%
    filter(video_count > threshold) %>%
    group_by(category_id, category) %>%
    summarise(videos_above_threshold = n(), .groups = "drop") %>%
    complete(category_id, category = dimension_order, fill = list(videos_above_threshold = 0)) %>%
    mutate(category = factor(category, levels = dimension_order))
  
  # Create visualization
  prevalence_plot <- ggplot(dimension_prevalence, 
                           aes(x = factor(category_id), y = videos_above_threshold, 
                               group = category, color = category)) +
    geom_line(size = 1) +
    geom_point(size = 2) +
    scale_color_brewer(palette = "Set2") +
    theme_classic() +
    labs(
      title = paste0("Dimension Appearing More Than ", threshold, " Times"),
      x = "Word Position",
      y = "Number of Videos",
      color = "Dimension"
    ) +
    theme(
      legend.position = "right",
      panel.grid.minor = element_blank(),
      axis.text.x = element_text(angle = 0)
    )
  
  return(list(data = dimension_prevalence, plot = prevalence_plot))
}

# Calculate and visualize dimension prevalence with threshold of 0 (any appearance)
dimension_prevalence_results <- calculate_dimension_prevalence(data, dimension_order, threshold = 0)
print(dimension_prevalence_results$plot)

# Calculate and visualize dimension prevalence with threshold of 40
dimension_prevalence_40_results <- calculate_dimension_prevalence(data, dimension_order, threshold = 40)
print(dimension_prevalence_40_results$plot)

# Print the data table for threshold 40
print("Number of videos with each dimension appearing more than 40 times at each position:")
print(dimension_prevalence_40_results$data)
