package com.ysz.dm.duitang.srv.support.blog.port.adapter.adapter;

import com.ysz.dm.duitang.srv.support.blog.domain.adapter.BlogCountAdapter;
import com.ysz.dm.duitang.srv.support.blog.domain.adapter.BlogCount;
import java.util.Collection;
import java.util.Map;

public class BlogCountAdapterImpl implements BlogCountAdapter {

  @Override
  public boolean incrBlogFavCount(final long blogId, final int step) {
    return false;
  }

  @Override
  public boolean decrBlogFavCount(final long blogId, final int step) {
    return false;
  }

  @Override
  public Map<Long, BlogCount> queryCountByIds(final Collection<Long> blogIds) {
    return null;
  }
}
