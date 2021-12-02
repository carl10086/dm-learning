package com.ysz.biz.tools.mysql.usage;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;

/**
 * <pre>
 *   比较 mysql.conf 文件显示所有不同的配置项.
 *
 *   输入:
 *    mysql -e"show variables" > local.conf 结果 .
 * </pre>
 */
public class MysqlConfDiff {

  private MysqlConf read(String filePath, String name) {
    final List<String> confs = readLines(filePath);
    final MysqlConf conf = new MysqlConf(name, confs.size());
    confs.forEach(x -> conf.addConfItem(x));
    return conf;
  }

  private List<String> readLines(String filePath) {
    try {
      return FileUtils.readLines(new File(filePath), Charset.defaultCharset());
    } catch (IOException e) {
      throw new RuntimeException("文件读取失败:" + filePath);
    }
  }

  public void execute(String jdFilePath, String dtFilePath) {
    final MysqlConf jd = read(jdFilePath, "jd");
    final MysqlConf dt = read(dtFilePath, "dt");

    List<String> res = new ArrayList<>();
    res.add("配置key,京东配置,堆糖配置");
    for (Entry<String, String> entry : jd.dict.entrySet()) {
      final String key = entry.getKey();
      /*暂时不考虑toku db*/
      if (key.contains("toku")) {
        continue;
      }
      String jdValue = entry.getValue();
      if (jdValue == null) {
        jdValue = StringUtils.EMPTY;
      }
      String dtValue = dt.dict.getOrDefault(key, StringUtils.EMPTY);

      if (!Objects.equals(jdValue, dtValue)) {
        res.add(String.format("\"%s\",\"%s\",\"%s\"", key, jdValue, dtValue));
      }
    }
    System.err.println(res.size());
    res.forEach(System.err::println);
  }


  public static void main(String[] args) throws Exception {
    new MysqlConfDiff().execute("/Users/carl/tmp/fuck/jd.conf", "/Users/carl/tmp/fuck/now.conf");
  }


  private static class MysqlConf {

    private String name;
    private Map<String, String> dict;

    public MysqlConf(String name, int initSize) {
      this.name = name;
      this.dict = new HashMap<>(initSize);
    }

    private void addConfItem(String line) {
      if (StringUtils.isNotBlank(line)) {
        final int length = line.length();

        try {
          final int index = line.indexOf('\t');
          if (index >= 0) {
            this.dict.put(
                line.substring(0, index).trim(),
                line.substring(index + 1).trim()
            );
          } else {
            this.dict.put(line.trim(), StringUtils.EMPTY);
          }
        } catch (Exception e) {
          throw new RuntimeException("解析失败:" + line, e);
        }

      }
    }
  }

}


