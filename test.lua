function sortedKeys(query, sortFunction)
  local keys, len = {}, 0
  for k,_ in pairs(query) do
    len = len + 1
    keys[len] = k
  end
  table.sort(keys, sortFunction)
  return keys
end

local query = {}
query['a'] = 1
query['b'] = 2
query['c'] = 3
query['d'] = 4
query['e'] = 5
query['f'] = 6
query['g'] = 7
query['h'] = 8
for _,k in pairs(sortedKeys(query)) do
  query[k]={}
end
print(query)