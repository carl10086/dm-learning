local cur_key = ARGV[2] .. ARGV[3]
local cur_count = redis.call("HGET", cur_key, "cur")
if not cur_count then
  local base_count = 0

  for i = 1, tonumber(ARGV[4]) - 1 do
    local prev_key = ARGV[2] .. tostring(tonumber(ARGV[3]) - i)
    local prev_cur = redis.call("HGET", prev_key, "cur")
    if prev_cur then
      base_count = base_count + prev_cur
    end
  end
  if base_count >= tonumber(ARGV[1]) then
    redis.call("HMSET", cur_key, "base", base_count, "cur", 0)
    redis.call("EXPIRE", cur_key, ARGV[5])
    return -base_count
  else
    redis.call("HMSET", cur_key, "base", base_count, "cur", 1)
    redis.call("EXPIRE", cur_key, ARGV[5])
    return base_count + 1
  end
else
  local total_cnt  = cur_count + tonumber(redis.call("HGET", cur_key, "base"))
  if total_cnt >= tonumber(ARGV[1]) then
    return -total_cnt
  else
    redis.call("HINCRBY", cur_key, "cur" , 1)
    return total_cnt + 1
  end
end