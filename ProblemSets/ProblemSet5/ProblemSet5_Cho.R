library(augsynth)
library(broom)
library(stringr)
library(readr)
library(xtable)
library(purrr)
library(tibble)
library(haven)
library(testthat)

# set up the directories
# creates output directories for figures and tables - OS-compatible paths
# @return list with paths to output directories

setup_directories <- function() {
  # relative paths that work across OS
  dirs <- list(
    figures = file.path("output", "figures"),
    tables = file.path("output", "tables"))
  
  # create directories if they don't exist
  for (dir in dirs) {
    dir.create(dir, showWarnings = FALSE, recursive = TRUE)}
  return(dirs)}

# (i) data cleaning function

#' clean the data 
#' processes raw Kansas tax data with treatment indicators
#' and pre-averaged covariates
#' 
#' @param kansas_raw data frame - raw Kansas economic data
#' @return list containing cleaned data frame (dfX)
#' @examples
#' \dontrun{
#' cleaned <- clean_kansas(kansas_raw_data)
#' }

# check whether input is in format of data frame
clean_kansas <- function(kansas_raw) {
  
  if (!is.data.frame(kansas_raw)) {
    stop("input should be in the data frame format in R")}
  
  required_cols <- c("year", "qtr", "gdp", "popestimate", "abb")
  if (!all(required_cols %in% names(kansas_raw))) {
    stop(paste("missing required columns:", 
               paste(setdiff(required_cols, names(kansas_raw)), collapse = ", ")))}
  
  df <- kansas_raw %>%
    dplyr::mutate(
      year_qtr = year + qtr/4 - 0.25,
      gdpcapita = gdp / popestimate * 1e6,
      revstatecapita = rev_state_total / popestimate * 1e6,
      revlocalcapita = rev_local_total / popestimate * 1e6,
      emplvlcapita = (month1_emplvl + month2_emplvl + month3_emplvl) / (3 * popestimate),
      totalwagescapita = total_qtrly_wages / popestimate,
      avgwklywagecapita = avg_wkly_wage,
      estabscapita = qtrly_estabs_count / popestimate) %>%
    dplyr::filter(!is.na(abb), abb != "DC") %>%
    dplyr::arrange(abb, year_qtr) %>%
    dplyr::mutate(
      Y = log(gdpcapita),
      D = as.integer(abb == "KS" & year_qtr >= 2012 + 1/4),
      unit = abb, time = year_qtr)
  
  # pre-period covariate averages
  pre_avgs <- df %>%
    dplyr::group_by(unit) %>%
    dplyr::summarize(
      pre_revstatecapita = mean(revstatecapita[time < 2012 + 1/4], na.rm = TRUE),
      pre_revlocalcapita = mean(revlocalcapita[time < 2012 + 1/4], na.rm = TRUE),
      pre_emplvlcapita = mean(emplvlcapita[time < 2012 + 1/4], na.rm = TRUE),
      pre_avgwklywagecapita = mean(avgwklywagecapita[time < 2012 + 1/4], na.rm = TRUE),
      pre_estabscapita = mean(estabscapita[time < 2012 + 1/4], na.rm = TRUE),.groups = "drop")
  
  # final dataset
  dfX <- df %>%
    dplyr::left_join(pre_avgs, by = "unit") %>%
    dplyr::arrange(unit, time) %>%
    dplyr::mutate(
      time = as.numeric(as.character(time)), D = as.integer(D))
  
  # validate output
  stopifnot(is.numeric(dfX$time), all(dfX$D %in% 0:1))
  return(list(dfX = dfX))}

# (ii) model fitting functions
#' fitting augmented synthetic control models
#' three variations of augmented synthetic control
#' ridge regularization, fixed effects augmentation, and generalized synthetic control (gen-SC)
#' 
#' @param dfX data frame with columns of Y,D,unit,time
#' @return list containing fitted models
#' @examples
#' \dontrun{
#' fits <- fit_models(cleaned_data$dfX)
#' }
fit_models <- function(dfX) {
  if (!is.data.frame(dfX)) {
    stop("dfX should be a data frame format in R")}
  required_cols <- c("Y", "D", "unit", "time")
  if (!all(required_cols %in% names(dfX))) {
    stop(paste("Missing required columns:", 
               paste(setdiff(required_cols, names(dfX)), collapse = ", ")))}
  
  # correct data types
  dfX <- dfX %>%
    dplyr::mutate(time = as.numeric(as.character(time)), D = as.integer(D))
  stopifnot(is.numeric(dfX$time), all(dfX$D %in% 0:1))
  
  # fit models with error handling
  fit_as_ridge <- tryCatch({
    augsynth::augsynth(
      Y ~ D, unit = unit, time = time, data = dfX, scm = TRUE, progfunc = "Ridge")}, error = function(e) {
        warning(paste("Ridge model failed:", e$message))
        NULL})
  
  fit_as_fe <- tryCatch({
    augsynth::augsynth(
      Y ~ D, unit = unit, time = time, data = dfX, scm = TRUE, progfunc = "None", fixedeff = TRUE)
  }, error = function(e) {
    warning(paste("Fixed effects model failed:", e$message))
    NULL})
  
  fit_as_gsyn <- NULL
  if (requireNamespace("gsynth", quietly = TRUE)) {
    fit_as_gsyn <- tryCatch({
      augsynth::augsynth(
        Y ~ D, unit = unit, time = time, data = dfX, scm = TRUE, progfunc = "GSYN")}, error = function(e) {
          warning(paste("GSYN model failed:", e$message))
          NULL})}
  
  return(list(
    fit_as_ridge = fit_as_ridge,
    fit_as_fe = fit_as_fe,
    fit_as_gsyn = fit_as_gsyn,
    dfX_checked = dfX))}

# (iii) visualisation function
#' plots for augmented synthetic control results - show the trends
#' 
#' @param dfX df for modeling
#' @param fits list of fitted models from fit_models()
#' @param outdir Output directory for plots (relative path)
#' @return list of files created
#' @examples
#' \dontrun{
#' make_visuals(data, fitted_models, file.path("output", "figures"))}
make_visuals <- function(dfX, fits, outdir = file.path("output", "figures")) {
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  files_created <- character()
  
  # save plots in OS-compatible way
  save_tiff_any <- function(filename, plot_expr, title_text = NULL, width = 6, height = 4, dpi = 300) {
    filepath <- file.path(outdir, filename)
    
    tryCatch({
      grDevices::tiff(
        filename = filepath,
        width = width, height = height, units = "in",
        res = dpi, compression = "lzw")
      op <- par(no.readonly = TRUE)
      on.exit({ 
        try(par(op), silent = TRUE)
        try(grDevices::dev.off(), silent = TRUE) 
      }, add = TRUE)
      
      par(mar = c(4.2, 4.8, 3.6, 1.5))
      
      obj <- eval.parent(substitute(plot_expr))
      
      if (inherits(obj, "ggplot")) {
        if (!is.null(title_text)) obj <- obj + ggplot2::ggtitle(title_text)
        print(obj)
      } else {
        if (!is.null(title_text)) {
          graphics::title(main = title_text)}}
      
      files_created <<- c(files_created, filepath)
      return(TRUE)
    }, error = function(e) {
      warning(paste("Failed to create", filename, ":", e$message))
      return(FALSE)})}
  
  event_title <- "2012 Kansas Tax Cuts on log(GDP)"
  if (!is.null(fits$fit_as_ridge)) {
    save_tiff_any(
      "fig1_ascm_ridge.tiff",
      plot_expr = { graphics::plot(fits$fit_as_ridge, type = "counterfactual") },
      title_text = paste0(event_title, " — ASCM (Ridge)"))}
  
  if (!is.null(fits$fit_as_fe)) {
    save_tiff_any(
      "fig2_ascm_fixedeff_gap.tiff",
      plot_expr = { graphics::plot(fits$fit_as_fe, type = "gap") },
      title_text = paste0(event_title, " — ASCM (Fixed Effects): Gap"))}
  
  if (!is.null(fits$fit_as_gsyn)) {
    save_tiff_any(
      "fig3_ascm_gsyn.tiff",
      plot_expr = { plot(fits$fit_as_gsyn, type = "counterfactual") },
      title_text = paste0(event_title, " — ASCM (GSYN)"))}
  return(invisible(files_created))}






###### att inference part ######
att_post_mean <- function(fit, dfX) {
  if (is.null(fit)) return(NA_real_)
  
  sm <- suppressWarnings(try(summary(fit), silent = TRUE))
  if (inherits(sm, "try-error")) return(NA_real_)
  
  # Try standard summary fields
  for (nm in c("avg_att", "att.avg", "att_avg", "ATT.avg", "ATT_avg")) {
    if (!is.null(sm[[nm]])) {
      return(as.numeric(sm[[nm]]))}}
  
  # Calculate from att vector if needed
  t0 <- min(dfX$time[dfX$D == 1], na.rm = TRUE)
  att_vec <- suppressWarnings(as.numeric(unlist(sm$att)))
  if (!length(att_vec)) return(NA_real_)
  
  n_post <- sum(dfX$time >= t0 & dfX$D %in% 0:1) / length(unique(dfX$unit))
  n_post <- max(1, floor(n_post))
  
  mean(tail(att_vec, n_post), na.rm = TRUE)
}

placebo_inference <- function(dfX, refitter, max_placebos = 30, seed = 123) {
  set.seed(seed)
  
  t0 <- min(dfX$time[dfX$D == 1], na.rm = TRUE)
  treated_unit <- unique(dfX$unit[dfX$D == 1])
  donors <- setdiff(unique(dfX$unit), treated_unit)
  
  if (length(donors) > max_placebos) {
    donors <- sample(donors, max_placebos)
  }
  
  # Get observed ATT
  fit_obs <- refitter(dfX)
  tau_hat <- att_post_mean(fit_obs, dfX)
  
  # Compute placebo ATTs
  placebo_atts <- vapply(donors, function(u) {
    dfP <- dfX
    dfP$D <- 0L
    dfP$D[dfP$unit == u & dfP$time >= t0] <- 1L
    
    fitP <- suppressWarnings(try(refitter(dfP), silent = TRUE))
    if (inherits(fitP, "try-error")) return(NA_real_)
    
    att_post_mean(fitP, dfP)
  }, numeric(1))
  
  placebo_atts <- placebo_atts[is.finite(placebo_atts)]
  
  # Calculate statistics
  se <- if (length(placebo_atts) > 1) stats::sd(placebo_atts) else NA_real_
  ci <- if (is.finite(tau_hat) && is.finite(se)) {
    c(tau_hat - 1.96*se, tau_hat + 1.96*se)
  } else {
    c(NA_real_, NA_real_)
  }
  
  pval <- if (length(placebo_atts)) {
    (1 + sum(abs(placebo_atts) >= abs(tau_hat))) / (length(placebo_atts) + 1)
  } else {
    NA_real_
  }
  
  return(list(
    att = tau_hat, 
    se = se, 
    ci_low = ci[1], 
    ci_high = ci[2], 
    p = pval,
    n_placebo = length(placebo_atts)
  ))
}

# unit test
#' Run unit tests for all functions
#' 
#' @return Test results
run_tests <- function() {
  test_that("setup_directories creates directories", {
    dirs <- setup_directories()
    expect_true(dir.exists(dirs$figures))
    expect_true(dir.exists(dirs$tables))})
  
  test_that("clean_kansas validates input", {
    expect_error(clean_kansas("not_a_dataframe"))
    expect_error(clean_kansas(data.frame(a = 1:10)))})
  
  test_that("fit_models handles missing columns", {
    bad_df <- data.frame(x = 1:10, y = 1:10)
    expect_error(fit_models(bad_df))})
  
  test_that("att_post_mean handles NULL input", {
    result <- att_post_mean(NULL, data.frame())
    expect_true(is.na(result))})
  
  test_that("make_visuals returns file list", {
    test_df <- data.frame(
      Y = rnorm(100),
      D = rep(0:1, each = 50),
      unit = rep(1:10, each = 10),
      time = rep(1:10, 10))
    
    test_fits <- list(
      fit_as_ridge = NULL,
      fit_as_fe = NULL,
      fit_as_gsyn = NULL,
      dfX_checked = test_df
    )
    
    # test function runs without error
    result <- make_visuals(test_df, test_fits, tempdir())
    expect_true(is.character(result) || is.null(result))})
  
  test_that("file paths are OS-compatible", {
    path <- file.path("output", "figures", "test.png")
    expect_true(grepl("output", path))
    # Should work on any OS
    expect_false(grepl("\\\\", path) && grepl("/", path))
  })
  cat("All tests completed\n")}

# main execution 
# runs the complete augmented synthetic control analysis
#' @param kansas_data input data frame
#' @return list with results and output paths

main_analysis <- function(kansas_data) {
  dirs <- setup_directories()
  cleaned <- clean_kansas(kansas_data)
  fits <- fit_models(cleaned$dfX)
  plot_files <- make_visuals(fits$dfX_checked, fits, dirs$figures)
  results_table <- summarize_models(fits)
  
  # save table
  output_files <- list(
    plots = plot_files,
    table_csv = file.path(dirs$tables, "results_table.csv"),
    table_tex = file.path(dirs$tables, "results_table.tex"))
  
  readr::write_csv(results_table, output_files$table_csv)
  write_latex_table(results_table, output_files$table_tex)
  
  return(list(
    results = results_table,
    files = output_files,
    fits = fits))}

if (interactive()) {
  cat("running the unit test now")
  run_tests()}

