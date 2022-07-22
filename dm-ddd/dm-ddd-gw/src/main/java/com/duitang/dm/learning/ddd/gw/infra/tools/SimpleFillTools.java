package com.duitang.dm.learning.ddd.gw.infra.tools;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class SimpleFillTools {


  public static <FROM, TO> List<TO> mapOnceAsList(
      Collection<FROM> fromCollection,
      Function<FROM, TO> function

  ) {
    if (fromCollection.isEmpty()) {
      return Collections.emptyList();
    }
    final List<TO> toList = new ArrayList<>(fromCollection.size());
    Stream<TO> toStream = fromCollection.stream().map(function);
    return toStream.filter(Objects::nonNull).collect(Collectors.toCollection(() -> toList));
  }


  public static <KEY, VALUE> Map<KEY, VALUE> mapOnceAsMap(
      Collection<VALUE> fromCollection,
      Function<VALUE, KEY> keyExtractor
  ) {
    if (fromCollection.isEmpty()) {
      return Collections.emptyMap();
    }
    Stream<VALUE> toStream = fromCollection.stream();
    return toStream.collect(Collectors.toMap(
        keyExtractor,
        value -> null,
        (value, value2) -> value2,
        (Supplier<Map<KEY, VALUE>>) () -> new HashMap<>(fromCollection.size())
    ));
  }

}
