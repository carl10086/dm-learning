local cur_key = ARGV[1] .. ARGV[2]
local count = redis.call("GET", cur_key)
if not count then
    redis.call("SETEX", cur_key, ARGV[3], "1")
    return 1
elseif tonumber(count) >= tonumber(ARGV[4]) then
    return -count
else
    local prev_count = redis.call("GET", ARGV[1] .. tostring(tonumber(ARGV[2]) - 1) )
    if prev_count then
        local prev_percentage = tonumber(ARGV[5]) + 0.0
        local total_count = (prev_percentage * prev_count) + count
        if  total_count >= tonumber(ARGV[4]) then
          return -total_count
        end
    end
    return redis.call("INCR", cur_key)
end
