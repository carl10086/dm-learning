package com.duitang.dm.learning.ddd.gw.domain.core.content;

import com.duitang.dm.learning.ddd.gw.infra.core.BaseCnt;
import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class UserContentCnt {


  private Long userId;

  private BaseCnt pubAtlasCnt;

  private BaseCnt pubAlbumAt;

  private BaseCnt forwardAtlasCnt;


  public UserContentCnt setUserId(Long userId) {
    this.userId = userId;
    return this;
  }

  public UserContentCnt setPubAtlasCnt(BaseCnt pubAtlasCnt) {
    this.pubAtlasCnt = pubAtlasCnt;
    return this;
  }

  public UserContentCnt setPubAlbumAt(BaseCnt pubAlbumAt) {
    this.pubAlbumAt = pubAlbumAt;
    return this;
  }

  public UserContentCnt setForwardAtlasCnt(BaseCnt forwardAtlasCnt) {
    this.forwardAtlasCnt = forwardAtlasCnt;
    return this;
  }
}
