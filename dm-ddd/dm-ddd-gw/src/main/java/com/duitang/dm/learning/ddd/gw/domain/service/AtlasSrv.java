package com.duitang.dm.learning.ddd.gw.domain.service;

import static com.duitang.dm.learning.ddd.gw.infra.tools.SimpleFillTools.mapOnceAsList;
import static com.duitang.dm.learning.ddd.gw.infra.tools.SimpleFillTools.mapOnceAsMap;

import com.duitang.dm.learning.ddd.gw.domain.aggregate.AtlasTinyAggregate;
import com.duitang.dm.learning.ddd.gw.domain.core.atlas.AtlasCore;
import com.duitang.dm.learning.ddd.gw.domain.core.atlas.AtlasCount;
import com.duitang.dm.learning.ddd.gw.domain.core.user.UserCore;
import java.util.List;
import java.util.Map;

public class AtlasSrv {

  public List<AtlasTinyAggregate> aggTiny(List<Long> atlasIds) {
    final List<AtlasCore> atlasCores = queryAtlasCores(atlasIds);

    List<AtlasTinyAggregate> atlasTinyAggregates = mapOnceAsList(atlasCores, AtlasTinyAggregate::new);
    fillAtlasTinyAggregate(atlasTinyAggregates);
    return atlasTinyAggregates;
  }

  private List<AtlasCore> queryAtlasCores(List<Long> atlasIds) {
    return null;
  }

  private void fillAtlasTinyAggregate(
      List<AtlasTinyAggregate> atlasTinyAggregates
  ) {
    if (atlasTinyAggregates.size() > 0) {
      final List<Long> atlasIds = mapOnceAsList(atlasTinyAggregates, AtlasTinyAggregate::id);

      final Map<Long, AtlasCount> atlasCountMap = mapOnceAsMap(queryAtlasCount(atlasIds), AtlasCount::getId);
      final Map<Long, UserCore> userCoreMap = mapOnceAsMap(queryUserCores(atlasIds), UserCore::getId);

      for (AtlasTinyAggregate atlasTinyAggregate : atlasTinyAggregates) {
        atlasTinyAggregate.setAtlasCount(atlasCountMap.get(atlasTinyAggregate.id()));
        atlasTinyAggregate.setUserCore(userCoreMap.get(atlasTinyAggregate.userId()));
      }
    }
  }

  private List<UserCore> queryUserCores(List<Long> userIds) {
    return null;
  }

  private List<AtlasCount> queryAtlasCount(List<Long> atlasIds) {
    return null;
  }
}
