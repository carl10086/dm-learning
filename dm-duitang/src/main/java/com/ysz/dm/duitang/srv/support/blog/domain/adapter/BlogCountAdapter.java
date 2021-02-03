package com.ysz.dm.duitang.srv.support.blog.domain.adapter;

import java.util.Collection;
import java.util.Map;

/**
 * 防腐层、比如说 likeCnt , favCnt 可能来自于不同的服务、这里负责 解耦、重新聚合 .
 */
public interface BlogCountAdapter {

  boolean incrBlogFavCount(final long blogId, final int step);

  boolean decrBlogFavCount(final long blogId, final int step);


  Map<Long, BlogCount> queryCountByIds(Collection<Long> blogIds);
}
