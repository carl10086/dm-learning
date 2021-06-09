//package com.ysz.dm.fast.basic.graph;
//
//
//import java.awt.AlphaComposite;
//import java.awt.Color;
//import java.awt.Font;
//import java.awt.Graphics2D;
//import java.awt.RenderingHints;
//import java.awt.image.BufferedImage;
//import java.io.File;
//import java.io.FileOutputStream;
//import java.io.IOException;
//import java.io.OutputStream;
//import javax.imageio.ImageIO;
//import sun.font.FontDesignMetrics;
//
//
//public class GraphDm {
//
//
//  public static int getWordWidth(Font font, String content) {
//    FontDesignMetrics metrics = FontDesignMetrics.getMetrics(font);
//    int width = 0;
//    for (int i = 0; i < content.length(); i++) {
//      width += metrics.charWidth(content.charAt(i));
//    }
//    return width;
//  }
//
//  public static void write(BufferedImage bufferedImage, String target) throws IOException {
//    File file = new File(target);
//    if (!file.getParentFile().exists()) {
//      file.getParentFile().mkdirs();
//    }
//    try (OutputStream os = new FileOutputStream(target)) {
//      ImageIO.write(bufferedImage, "PNG", os);
//    }
//  }
//
//
//  public static void main(String[] args) throws Exception {
//    Font font = new Font("微软雅黑", Font.BOLD, 32);
//    String content = "你好Java!";
//    FontDesignMetrics metrics = FontDesignMetrics.getMetrics(font);
//    int width = getWordWidth(font, content);//计算图片的宽
//    int height = metrics.getHeight();//计算高
//    BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
//    Graphics2D graphics = bufferedImage.createGraphics();
//    graphics.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING,
//        RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
//    graphics.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER));
//    //设置背影为白色
//    graphics.setColor(Color.WHITE);
//    graphics.fillRect(0, 0, bufferedImage.getWidth(), bufferedImage.getHeight());
//    graphics.setFont(font);
//    graphics.setColor(Color.BLACK);
//    graphics.drawString(content, 0, metrics.getAscent());//图片上写文字
//    graphics.dispose();
//    write(bufferedImage, "/data/test.png");
//
//  }
//
//}
