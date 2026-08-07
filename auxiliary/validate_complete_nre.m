function curve = validate_complete_nre(curve, expected_length, algorithm_name, trial_index)
%VALIDATE_COMPLETE_NRE Validate one complete per-frame NRE trajectory.

if nargin < 4
    trial_index = NaN;
end
if nargin < 3 || isempty(algorithm_name)
    algorithm_name = 'Unknown algorithm';
end
if ~(isnumeric(expected_length) && isreal(expected_length) && ...
        isscalar(expected_length) && isfinite(expected_length) && ...
        expected_length >= 1 && expected_length == floor(expected_length))
    error('validate_complete_nre:InvalidExpectedLength', ...
        'The expected NRE length must be a positive integer.');
end

algorithm_name = char(algorithm_name);
trial_text = mat2str(trial_index);

if ~(isnumeric(curve) && isreal(curve) && isvector(curve))
    error('validate_complete_nre:InvalidType', ...
        '%s returned a non-real or non-vector NRE output in trial %s.', ...
        algorithm_name, trial_text);
end

curve = reshape(curve, 1, []);
if numel(curve) ~= expected_length
    error('validate_complete_nre:InvalidLength', ...
        '%s returned %d NRE values in trial %s; %d values were expected.', ...
        algorithm_name, numel(curve), trial_text, expected_length);
end

first_invalid = find(~isfinite(curve), 1, 'first');
if ~isempty(first_invalid)
    error('validate_complete_nre:NonfiniteValue', ...
        '%s returned a nonfinite NRE at frame %d in trial %s.', ...
        algorithm_name, first_invalid, trial_text);
end

first_negative = find(curve < 0, 1, 'first');
if ~isempty(first_negative)
    error('validate_complete_nre:NegativeValue', ...
        '%s returned a negative NRE at frame %d in trial %s.', ...
        algorithm_name, first_negative, trial_text);
end
end
