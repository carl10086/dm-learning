package com.duitang.dm.learning.ddd.gw.domain.service;

import static com.duitang.dm.learning.ddd.gw.infra.tools.SimpleFillTools.mapOnceAsList;
import static com.duitang.dm.learning.ddd.gw.infra.tools.SimpleFillTools.mapOnceAsMap;

import com.duitang.dm.learning.ddd.gw.domain.assemble.AtlasTinyAssemble;
import com.duitang.dm.learning.ddd.gw.domain.core.content.atlas.AtlasCore;
import com.duitang.dm.learning.ddd.gw.domain.core.uact.BizObjectCount;
import com.duitang.dm.learning.ddd.gw.domain.core.user.UserCore;
import java.util.List;
import java.util.Map;

public class AtlasSrv {

  public List<AtlasTinyAssemble> queryTinyAtlasAggs(List<Long> atlasIds) {
    final List<AtlasCore> atlasCores = queryAtlasCores(atlasIds);

    List<AtlasTinyAssemble> atlasTinyAssembles = mapOnceAsList(atlasCores, AtlasTinyAssemble::new);
    fillAtlasTinyAggregate(atlasTinyAssembles);
    return atlasTinyAssembles;
  }

  private List<AtlasCore> queryAtlasCores(List<Long> atlasIds) {
    return null;
  }

  private void fillAtlasTinyAggregate(
      List<AtlasTinyAssemble> atlasTinyAssembles
  ) {
    if (atlasTinyAssembles.size() > 0) {
      final List<Long> atlasIds = mapOnceAsList(atlasTinyAssembles, AtlasTinyAssemble::id);

      final Map<Long, BizObjectCount> atlasCountMap = mapOnceAsMap(queryAtlasCount(atlasIds), BizObjectCount::getId);
      final Map<Long, UserCore> userCoreMap = mapOnceAsMap(queryUserCores(atlasIds), UserCore::getId);

      for (AtlasTinyAssemble atlasTinyAssemble : atlasTinyAssembles) {
        atlasTinyAssemble.setBizObjectCount(atlasCountMap.get(atlasTinyAssemble.id()));
        atlasTinyAssemble.setAuthor(userCoreMap.get(atlasTinyAssemble.userId()));
      }
    }
  }

  private List<UserCore> queryUserCores(List<Long> userIds) {
    return null;
  }

  private List<BizObjectCount> queryAtlasCount(List<Long> atlasIds) {
    return null;
  }
}
