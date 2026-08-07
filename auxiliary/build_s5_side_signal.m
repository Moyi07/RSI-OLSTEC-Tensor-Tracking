function [side_signal, effective_signal, threshold, realized_delay, ...
    missing_rate, onset_gradient_to_threshold] = build_s5_side_signal( ...
    condition, transition_boundary, total_frames, side_seed, config)
%BUILD_S5_SIDE_SIGNAL Generate one controlled S5 side-information stream.
% Event delays are measured from the first post-mutation frame.

    true_event = zeros(total_frames, 1);
    true_event_start = NaN;
    realized_delay = NaN;
    if condition.include_true_event
        realized_delay = condition.fixed_event_offset;
        if ~isempty(condition.random_delay_range)
            stream = RandStream('mt19937ar', 'Seed', side_seed + 20000);
            realized_delay = randi(stream, ...
                condition.random_delay_range, 1, 1);
        end
        true_event_start = transition_boundary + realized_delay;
        true_event = make_event(total_frames, true_event_start, ...
            config.event_amplitude, config.event_decay);
    end

    false_event = zeros(total_frames, 1);
    if condition.false_event_gain > 0
        false_event_start = ...
            transition_boundary + condition.false_event_offset;
        false_event = make_event(total_frames, false_event_start, ...
            config.event_amplitude, config.event_decay);
    end

    event_component = condition.event_gain * true_event + ...
        condition.false_event_gain * false_event;
    stream = RandStream('mt19937ar', 'Seed', side_seed);
    standardized_noise = randn(stream, total_frames, 1);
    side_signal = config.baseline + event_component + ...
        condition.noise_sigma * standardized_noise;

    if condition.smoothing_beta > 0
        beta = condition.smoothing_beta;
        for frame = 2:total_frames
            side_signal(frame) = beta * side_signal(frame - 1) + ...
                (1 - beta) * side_signal(frame);
        end
    end

    missing_mask = false(total_frames, 1);
    tracking_frames = (config.burn_in_frames + 1):total_frames;
    switch condition.missing_mode
        case 'random'
            stream = RandStream('mt19937ar', 'Seed', side_seed + 10000);
            draws = rand(stream, numel(tracking_frames), 1);
            missing_mask(tracking_frames) = draws < condition.missing_rate;
        case 'burst'
            first_missing = transition_boundary;
            last_missing = first_missing + ...
                condition.missing_burst_length - 1;
            missing_mask(first_missing:last_missing) = true;
    end
    side_signal(missing_mask) = NaN;

    calibration_difference = ...
        diff(side_signal(1:config.burn_in_frames));
    center = median(calibration_difference);
    mad_value = median(abs(calibration_difference - center));
    estimated_sigma = (1.4826 * mad_value) / sqrt(2);
    threshold = max(config.threshold_floor, ...
        config.threshold_sigma_multiplier * sqrt(2) * estimated_sigma);

    effective_signal = side_signal;
    for frame = 2:total_frames
        if isnan(effective_signal(frame))
            effective_signal(frame) = effective_signal(frame - 1);
        end
    end

    missing_rate = nnz(missing_mask(tracking_frames)) / ...
        numel(tracking_frames);
    if isfinite(true_event_start)
        event_gradient = abs(effective_signal(true_event_start) - ...
            effective_signal(true_event_start - 1));
        onset_gradient_to_threshold = event_gradient / threshold;
    else
        onset_gradient_to_threshold = NaN;
    end
end

function event = make_event(total_frames, start_frame, amplitude, decay)
    if start_frame < 2 || start_frame > total_frames
        error('build_s5_side_signal:InvalidEventTime', ...
            'The side-information event lies outside the sequence.');
    end
    event = zeros(total_frames, 1);
    event(start_frame) = amplitude;
    for frame = start_frame + 1:total_frames
        event(frame) = decay * event(frame - 1);
    end
end
