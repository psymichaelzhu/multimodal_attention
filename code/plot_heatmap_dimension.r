# objective ====
# todo:
# - lemmatization 
# - map to categories
# - select relevant columns
# findings: 
# - same pattern holds for different n_word X n_simulation combinations
#   - high n_simulation -> rare cases amplified (e.g. camouflage)

# preparation ====
# packages 
library(tidyverse)
library(stringr)
library(grid)
library(gridExtra)
library(png)

# load data
# data parameters
frame_index <- 40 # which frame is annotated
n_word <- 10 # how many relevant words are allowed
n_simulation <- 50 # how many times the annotation is repeated

# visualization parameters
n_unique_word <- 50 # exhibit the top `n_unique_word` most frequent words
cut_word <- 10 # if integer, exhibit the first `cut_word` positions


frame_index_str <- paste0("frame", frame_index, "_word", n_word, "_sim", n_simulation)
data <- read_csv(paste0("/Users/rezek_zhu/multimodal_attention/data/video/gpt_free-association/processed_", frame_index_str, ".csv"))

# data processing ====
video_names <- data %>%
  distinct(video_name) %>%
  pull(video_name)

for (video_index in 21:50) {
  # split free-association content into words
  data_long <- data %>%
    filter(video_name %in% video_names[video_index]) %>% 
  select(video_name, simulation_id, starts_with("category_")) %>% 
  pivot_longer(
    cols = starts_with("category_"),
    names_to = "category_id",
    values_to = "category"
  )%>% 
  mutate(category_id = as.numeric(str_extract(category_id, "\\d+"))) %>%
  filter(category_id <= cut_word) %>%
  filter(category != "not_found")  # Remove not_found categories

# visualization: category frequencies ====
by_position <- data_long %>%
  group_by(category_id, category) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(category_id) %>%
  mutate(proportion = n/sum(n)) %>%
  arrange(category_id, desc(proportion)) %>%
  group_by(category_id) %>%
  slice_head(n = n_unique_word)

overall_freq <- data_long %>%
  group_by(category) %>%
  summarise(n = n()) %>%
  mutate(proportion = n/sum(n)) %>%
  arrange(desc(proportion)) %>%
  slice_head(n = n_unique_word)

position_plots <- ggplot(by_position, aes(x = reorder(category, proportion), y = proportion)) +
  geom_bar(stat = "identity") +
  facet_wrap(~category_id, scales = "free_y") +
  coord_flip() +
  theme_minimal() +
  labs(x = "Category", y = "Proportion")

overall_plot <- ggplot(overall_freq, aes(x = reorder(category, proportion), y = proportion)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(x = "Category", y = "Proportion", title = "Overall Category Frequencies")

#print(position_plots)
#print(overall_plot)


# visualization: top categories by position ====
# Define consistent dimension order
dimension_order <- c("character", "location", "activity", "object", "relationship", "emotion_state", "other") %>% rev()

# Filter categories to include only those in our dimension order
top_categories <- overall_freq %>%
  filter(category %in% dimension_order) %>%
  arrange(match(category, dimension_order)) %>%
  pull(category)

top_by_position <- data_long %>%
  filter(category %in% dimension_order) %>%
  group_by(category_id, category) %>%
  summarise(count = n(), .groups = "drop") %>%
  complete(category_id, category = dimension_order, fill = list(count = 0)) %>%
  mutate(category = factor(category, levels = dimension_order))

# Get the current video name for the frame image
current_video <- data_long %>% 
  distinct(video_name) %>% 
  pull(video_name)

# Load the corresponding frame image
frame_path <- paste0("/Users/rezek_zhu/multimodal_attention/data/video/frame/", frame_index, "/", current_video, ".png")
frame_img <- NULL
if (file.exists(frame_path)) {
  frame_img <- readPNG(frame_path)
} else {
  warning(paste("Frame image not found:", frame_path))
}

# Create the heatmap
heatmap_plot <- ggplot(top_by_position, aes(x = factor(category_id), y = category, fill = count)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "darkblue") +
  theme_minimal() +
  labs(x = "Position", y = "Category", fill = "Count") +
  theme(axis.text.x = element_text(angle = 0))

# Combine heatmap with frame image
if (!is.null(frame_img)) {
  # Create a ggplot for the image
  img_plot <- ggplot() + 
    annotation_custom(rasterGrob(frame_img, interpolate = TRUE), 
                      xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf) +
    theme_void() +
    labs(title = paste(current_video)) +
    theme(plot.title = element_text(hjust = 0.5))
  
  # Arrange the plots vertically
  combined_plot <- grid.arrange(heatmap_plot, img_plot, ncol = 1, 
                                heights = c(2, 1.5))
  
  # Display the combined plot
  print(combined_plot)
} else {
  # Just display the heatmap if image not found
  print(heatmap_plot)
}
}

# todo
# synthesize categories