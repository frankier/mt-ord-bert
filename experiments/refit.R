#!/usr/bin/Rscript

library(VGAM)
library(arrow)


main <- function(args) {
    if (args[1] == "cumulative_parallel") {
        model <- cumulative(parallel = TRUE)
    } else if (args[1] == "cumulative") {
        model <- cumulative(parallel = FALSE)
    } else if (args[1] == "acat_parallel") {
        model <- acat(parallel = TRUE)
    } else if (args[1] == "acat") {
        model <- acat(parallel = FALSE)
    } else {
        stop("Unknown model")
    }

    df <- read_parquet(args[2])
    task_ids <- sort(unique(df$task_ids))
    mat_names <- character(length(task_ids))

    for (task_id in task_ids) {
        task_df <- df[df$task_ids == task_id, ]
        scale_points <- task_df$scale_points[1]
        task_df$label <- factor(task_df$label + 1, levels=1:scale_points)
        for (level in 1:scale_points) {
            if (nrow(task_df[task_df$label == level, ]) > 0) {
                next
            }
            prev_idx <- NULL
            for (i in (level-1):1) {
                if (nrow(task_df[task_df$label == i, ]) > 0) {
                    prev_idx <- i
                    break
                }
            }
            next_idx <- NULL
            for (i in (level+1):scale_points) {
                if (nrow(task_df[task_df$label == i, ]) > 0) {
                    next_idx <- i
                    break
                }
            }
            if (is.null(prev_idx)) {
                fake_hidden <- min(task_df[task_df$label == next_idx, "hidden"])
            } else if (is.null(next_idx)) {
                fake_hidden <- max(task_df[task_df$label == prev_idx, "hidden"])
            } else {
                fake_hidden <- mean(c(
                    mean(task_df[task_df$label == prev_idx, "hidden"]),
                    mean(task_df[task_df$label == next_idx, "hidden"])
                ))
            }
            # task_ids, label, scale_points, hidden
            task_df[nrow(task_df) + 1, ] <- list(task_id, level, scale_points, fake_hidden)
        }
        fit_cumul_par <- vglm(label ~ hidden, model, data = task_df)
        mat <- coef(fit_cumul_par, matrix = TRUE)
        name <- paste("t", task_id, sep = "")
        assign(name, mat)
        mat_names[[task_id + 1]] <- name
    }

    save(list=mat_names, file=args[3])
}

if (sys.nframe() == 0){
    args = commandArgs(trailingOnly=TRUE)
    main(args)
}