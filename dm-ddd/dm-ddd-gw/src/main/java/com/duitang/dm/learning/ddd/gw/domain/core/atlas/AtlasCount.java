package com.duitang.dm.learning.ddd.gw.domain.core.atlas;

import java.time.Instant;
import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class AtlasCount {

  private Long id;

  private Integer favCnt;

  private Integer likeCnt;

  private Integer commentCnt;

  private Integer commentUserCnt;

  private Instant lastFavAt;

  private Instant lastCommentAt;

  private Instant lastLikeAt;

  public AtlasCount setId(Long id) {
    this.id = id;
    return this;
  }

  public AtlasCount setFavCnt(Integer favCnt) {
    this.favCnt = favCnt;
    return this;
  }

  public AtlasCount setLikeCnt(Integer likeCnt) {
    this.likeCnt = likeCnt;
    return this;
  }

  public AtlasCount setCommentCnt(Integer commentCnt) {
    this.commentCnt = commentCnt;
    return this;
  }

  public AtlasCount setCommentUserCnt(Integer commentUserCnt) {
    this.commentUserCnt = commentUserCnt;
    return this;
  }

  public AtlasCount setLastFavAt(Instant lastFavAt) {
    this.lastFavAt = lastFavAt;
    return this;
  }

  public AtlasCount setLastCommentAt(Instant lastCommentAt) {
    this.lastCommentAt = lastCommentAt;
    return this;
  }

  public AtlasCount setLastLikeAt(Instant lastLikeAt) {
    this.lastLikeAt = lastLikeAt;
    return this;
  }
}
