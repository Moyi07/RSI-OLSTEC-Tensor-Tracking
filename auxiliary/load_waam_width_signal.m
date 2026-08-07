function [aux_width, meta_info] = load_waam_width_signal(meta_filename, video_filename, total_slices, missing_policy)
%LOAD_WAAM_WIDTH_SIGNAL Load measured melt-pool width side information.
%   [aux_width, meta_info] = load_waam_width_signal(meta_filename,
%   video_filename, total_slices) reads Width_mm from WAMVID_metadata.csv,
%   matches the requested video by file name, trims leading unavailable
%   measurements, and causally forward-fills later missing entries.
%
%   No random fallback is used. Missing or malformed measured metadata raises
%   an error so real-data experiments cannot silently use synthetic side
%   information.

    if nargin < 3
        error('load_waam_width_signal:InvalidInput', ...
            'meta_filename, video_filename, and total_slices are required.');
    end
    if nargin < 4 || isempty(missing_policy)
        missing_policy = 'trim_leading_nan';
    end
    if ~exist(meta_filename, 'file')
        error('load_waam_width_signal:MissingMetadata', ...
            'Metadata file not found: %s', meta_filename);
    end
    if ~isscalar(total_slices) || total_slices < 1 || total_slices ~= floor(total_slices)
        error('load_waam_width_signal:InvalidSliceCount', ...
            'total_slices must be a positive integer scalar.');
    end

    meta_table = readtable(meta_filename, 'Delimiter', ',');
    names = meta_table.Properties.VariableNames;
    video_col_idx = find(strcmpi(names, 'Video_filepath'), 1);
    width_col_idx = find(strcmpi(names, 'Width_mm'), 1);
    if isempty(video_col_idx) || isempty(width_col_idx)
        error('load_waam_width_signal:MissingColumns', ...
            'Metadata must contain Video_filepath and Width_mm columns.');
    end

    video_col = meta_table{:, video_col_idx};
    width_col = meta_table{:, width_col_idx};
    target_base = local_filename(video_filename);

    is_match = false(height(meta_table), 1);
    for row = 1:height(meta_table)
        row_video = local_cell_to_char(video_col, row);
        row_base = local_filename(row_video);
        if strcmpi(row_base, target_base)
            is_match(row) = true;
        end
    end
    match_idx = find(is_match);
    if isempty(match_idx)
        error('load_waam_width_signal:VideoNotMatched', ...
            'No metadata row matched video file: %s', video_filename);
    end
    if numel(match_idx) > 1
        error('load_waam_width_signal:AmbiguousVideoMatch', ...
            'Multiple metadata rows matched video file: %s', video_filename);
    end

    row_idx = match_idx(1);
    width_raw = local_parse_numeric_array(local_cell_to_char(width_col, row_idx));
    if numel(width_raw) < total_slices
        error('load_waam_width_signal:ShortMetadata', ...
            'Width_mm has %d entries but %d video frames are requested.', ...
            numel(width_raw), total_slices);
    end
    width_raw = width_raw(1:total_slices);

    first_valid = find(isfinite(width_raw), 1, 'first');
    if isempty(first_valid)
        error('load_waam_width_signal:NoValidWidthInVideoWindow', ...
            'Width_mm contains no valid measured values in the requested video window.');
    end

    switch lower(char(missing_policy))
        case {'trim_leading_nan', 'strict'}
            aux_width = width_raw(first_valid:end);
            trim_start_frame = first_valid;
            num_trimmed_front = first_valid - 1;
        otherwise
            error('load_waam_width_signal:InvalidMissingPolicy', ...
                'missing_policy must be trim_leading_nan.');
    end

    num_forward_filled = 0;
    for i = 2:numel(aux_width)
        if ~isfinite(aux_width(i))
            aux_width(i) = aux_width(i-1);
            num_forward_filled = num_forward_filled + 1;
        end
    end
    if any(~isfinite(aux_width))
        error('load_waam_width_signal:UnfilledWidth', ...
            'Width_mm contains missing values that could not be causally filled.');
    end

    meta_info = struct();
    meta_info.row_idx = row_idx;
    meta_info.video_filepath = local_cell_to_char(video_col, row_idx);
    meta_info.missing_policy = missing_policy;
    meta_info.trim_start_frame = trim_start_frame;
    meta_info.num_trimmed_front = num_trimmed_front;
    meta_info.original_length = numel(width_raw);
    meta_info.effective_length = numel(aux_width);
    meta_info.num_forward_filled = num_forward_filled;
end

function out = local_cell_to_char(col, row)
    if iscell(col)
        value = col{row};
    else
        value = col(row);
    end

    if isstring(value)
        out = char(value);
    elseif ischar(value)
        out = value;
    elseif isnumeric(value)
        out = num2str(value);
    else
        out = char(value);
    end
end

function base = local_filename(path_text)
    path_text = strrep(char(path_text), '/', filesep);
    path_text = strrep(path_text, '\', filesep);
    [~, name, ext] = fileparts(path_text);
    base = [name, ext];
end

function values = local_parse_numeric_array(raw_text)
    raw_text = char(raw_text);
    tokens = regexp(raw_text, ...
        '(?i)nan|[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?', 'match');
    if isempty(tokens)
        error('load_waam_width_signal:MissingWidth', ...
            'Width_mm entry does not contain numeric values.');
    end

    values = NaN(numel(tokens), 1);
    for i = 1:numel(tokens)
        if strcmpi(tokens{i}, 'nan')
            values(i) = NaN;
        else
            values(i) = str2double(tokens{i});
        end
    end
end
