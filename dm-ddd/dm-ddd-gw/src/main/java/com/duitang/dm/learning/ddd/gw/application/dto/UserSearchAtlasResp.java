package com.duitang.dm.learning.ddd.gw.application.dto;

import com.duitang.dm.learning.ddd.gw.domain.aggregate.AtlasTinyAggregate;
import java.util.List;
import java.util.Map;
import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class UserSearchAtlasResp {

  private List<AtlasTinyAggregate> atlasTinyAggregates;
  private Map<Long, Boolean> atlasLikeMap;
  private Map<Long, Boolean> atlasFavMap;

  public UserSearchAtlasResp(
      List<AtlasTinyAggregate> atlasTinyAggregates,
      Map<Long, Boolean> atlasLikeMap,
      Map<Long, Boolean> atlasFavMap
  ) {
    this.atlasTinyAggregates = atlasTinyAggregates;
    this.atlasLikeMap = atlasLikeMap;
    this.atlasFavMap = atlasFavMap;
  }

  public UserSearchAtlasResp setAtlasTinyAggregates(List<AtlasTinyAggregate> atlasTinyAggregates) {
    this.atlasTinyAggregates = atlasTinyAggregates;
    return this;
  }

  public UserSearchAtlasResp setAtlasLikeMap(Map<Long, Boolean> atlasLikeMap) {
    this.atlasLikeMap = atlasLikeMap;
    return this;
  }

  public UserSearchAtlasResp setAtlasFavMap(Map<Long, Boolean> atlasFavMap) {
    this.atlasFavMap = atlasFavMap;
    return this;
  }
}
