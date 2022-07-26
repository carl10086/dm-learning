package com.duitang.dm.learning.ddd.gw.application.dto;

import com.duitang.dm.learning.ddd.gw.domain.assemble.AtlasTinyAssemble;
import java.util.List;
import java.util.Map;
import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class UserSearchAtlasResp {

  private List<AtlasTinyAssemble> atlasTinyAssembles;
  private Map<Long, Boolean> atlasLikeMap;
  private Map<Long, Boolean> atlasFavMap;

  public UserSearchAtlasResp(
      List<AtlasTinyAssemble> atlasTinyAssembles,
      Map<Long, Boolean> atlasLikeMap,
      Map<Long, Boolean> atlasFavMap
  ) {
    this.atlasTinyAssembles = atlasTinyAssembles;
    this.atlasLikeMap = atlasLikeMap;
    this.atlasFavMap = atlasFavMap;
  }

  public UserSearchAtlasResp setAtlasTinyAssembles(List<AtlasTinyAssemble> atlasTinyAssembles) {
    this.atlasTinyAssembles = atlasTinyAssembles;
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
