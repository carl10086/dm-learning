local count = redis.call("GET", KEYS[1])
if not count then
    redis.call("SETEX", KEYS[1], ARGV[1], "1")
    return 1
elseif tonumber(count) > tonumber(ARGV[2]) then
    return -1
else
    return redis.call("INCR", KEYS[1])
end
