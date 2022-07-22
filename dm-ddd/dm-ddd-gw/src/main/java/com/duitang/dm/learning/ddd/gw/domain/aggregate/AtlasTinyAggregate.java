package com.duitang.dm.learning.ddd.gw.domain.aggregate;

import com.duitang.dm.learning.ddd.gw.domain.core.atlas.AtlasCore;
import com.duitang.dm.learning.ddd.gw.domain.core.atlas.AtlasCount;
import com.duitang.dm.learning.ddd.gw.domain.core.user.UserCore;
import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class AtlasTinyAggregate {

  private AtlasCore atlasCore;
  private AtlasCount atlasCount;
  private UserCore userCore;

  public AtlasTinyAggregate(AtlasCore atlasCore) {
    this.atlasCore = atlasCore;
  }

  public Long userId() {
    return this.userCore.getId();
  }


  public Long id() {
    return this.atlasCore.getId();
  }

  public AtlasTinyAggregate setAtlasCore(AtlasCore atlasCore) {
    this.atlasCore = atlasCore;
    return this;
  }

  public AtlasTinyAggregate setAtlasCount(AtlasCount atlasCount) {
    this.atlasCount = atlasCount;
    return this;
  }

  public AtlasTinyAggregate setUserCore(UserCore userCore) {
    this.userCore = userCore;
    return this;
  }
}
