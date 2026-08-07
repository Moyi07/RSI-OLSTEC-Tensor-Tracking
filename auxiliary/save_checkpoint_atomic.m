function save_checkpoint_atomic(checkpoint_file, checkpoint)
%SAVE_CHECKPOINT_ATOMIC Save a checkpoint without exposing a partial file.

    temp_file = [checkpoint_file, '.tmp.mat'];
    if exist(temp_file, 'file')
        delete(temp_file);
    end

    save(temp_file, 'checkpoint', '-v7');
    [ok, message] = movefile(temp_file, checkpoint_file, 'f');
    if ~ok
        if exist(temp_file, 'file')
            delete(temp_file);
        end
        error('save_checkpoint_atomic:MoveFailed', ...
            'Failed to replace checkpoint %s: %s', checkpoint_file, message);
    end
end
