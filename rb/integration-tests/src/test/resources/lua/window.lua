local count = redis.call("GET", KEYS[1])
if not count then
    return redis.call("INCR", KEYS[1])
else
    redis.call("SETEX", KEYS[1], ARGV[1], "1")
    return 1
end
