function randperm(n)
  t={}
  for i=1,n do
    table.insert(t,i)
  end
  while n > 1 do
    local k = math.random(n)
    t[n], t[k] = t[k], t[n]
    n = n - 1
  end
  return t
end