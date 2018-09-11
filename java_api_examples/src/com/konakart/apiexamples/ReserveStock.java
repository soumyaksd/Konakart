//
// (c) 2015 DS Data Systems UK Ltd, All rights reserved.
//
// DS Data Systems and KonaKart and their respective logos, are 
// trademarks of DS Data Systems UK Ltd. All rights reserved.
//
// The information in this document is free software; you can redistribute 
// it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This software is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
package com.konakart.apiexamples;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.text.SimpleDateFormat;

import org.apache.commons.configuration.ConfigurationException;

import com.konakart.app.AddToBasketOptions;
import com.konakart.app.Basket;
import com.konakart.app.KKException;
import com.konakart.app.StockReservationOptions;
import com.konakart.appif.AddToBasketOptionsIf;
import com.konakart.appif.BasketIf;
import com.konakart.appif.OptionIf;
import com.konakart.appif.ProductIf;
import com.konakart.appif.StockReservationOptionsIf;

/**
 * This class shows how to call the KonaKart API to reserve stock. Before running you may have to
 * edit BaseApiExample.java to change the username and password used to log into the engine. The
 * default values are doe@konakart.com / password .
 */
public class ReserveStock extends BaseApiExample
{
    private static final String usage = "Usage: ReserveStock\n" + COMMON_USAGE;

    // Matrox MG400-32MB in default DB. A product with options.
    private static int matroxProdId;

    // Blade Runner DVD in default DB.
    private static int bladeRunnerProdId;

    /**
     * @param args
     */
    public static void main(String[] args)
    {
        try
        {
            ReserveStock myReserveStock = new ReserveStock();
            myReserveStock.reserveStock(args, usage);
        } catch (Exception e)
        {
            System.out.println("WARNING : There was a problem Reserving Stock");
            e.printStackTrace();
        }
    }

    /**
     * @param args
     *            command line arguments
     * @param use
     *            usage string
     * @throws IllegalAccessException
     * @throws InstantiationException
     * @throws ClassNotFoundException
     * @throws KKException
     * @throws ConfigurationException
     * @throws IOException
     * @throws InvocationTargetException
     * @throws IllegalArgumentException
     */
    public void reserveStock(String[] args, String use) throws KKException, ClassNotFoundException,
            InstantiationException, IllegalAccessException, ConfigurationException, IOException,
            IllegalArgumentException, InvocationTargetException
    {
        parseArgs(args, use, 0);

        /*
         * Get an instance of the KonaKart engine and login. The method called can be found in
         * BaseApiExample.java
         */
        init();

        /*
         * Create an array of basketItems which we send to the engine when attempting to reserve
         * stock.
         */
        matroxProdId = getProductIdByName("Matrox G200 MMS");
        bladeRunnerProdId = getProductIdByName("Blade Runner - Director's Cut");

        // Instantiate an AddToBasketOptions object
        AddToBasketOptionsIf basketOptions = new AddToBasketOptions();

        // Remove any existing basket items for this customer
        eng.removeBasketItemsPerCustomer(sessionId, -1);

        /*
         * Get the product with options. Note that the sessionId may be set to null since you don't
         * have to be logged in to get a product. However, if you are logged in, the engine will
         * calculate the correct price based on your location since the tax may vary based on
         * location. We would normally get the product from the DB to display to the customer and so
         * that the customer can choose the options (i.e. Shoe size).
         */
        ProductIf prod = eng.getProduct(sessionId, matroxProdId, DEFAULT_LANGUAGE);

        if (prod == null)
        {
            throw new KKException("Unexpected Problem : Could not find a product using Id "
                    + matroxProdId);
        }

        // Ensure that the product has options
        if (prod.getOpts() != null && prod.getOpts().length >= 3)
        {
            // Create a basket item
            BasketIf item = new Basket();

            // Create an OptionIf[] and add a couple of the available product options
            OptionIf[] opts = new OptionIf[2];
            opts[0] = prod.getOpts()[0];
            opts[1] = prod.getOpts()[2];

            // Set the product id for the basket item
            item.setProductId(matroxProdId);
            // Set the quantity of products to buy
            item.setQuantity(2);
            // Add the options
            item.setOpts(opts);

            // Add this basket item to the basket
            eng.addToBasketWithOptions(sessionId, 0, item, basketOptions);
        }

        /*
         * We add another product to the basket. This time with no options. Since we don't have to
         * determine the available options, we don't have to retrieve the product from the database
         * as long as we have the product id. In reality in a web application, we would retrieve it
         * since the customer would want to read up on the product before buying it.
         */
        // Create a basket item
        BasketIf item1 = new Basket();

        // Set the product id for the basket item
        item1.setProductId(bladeRunnerProdId);
        // Set the quantity of products to buy
        item1.setQuantity(1);

        // Add this basket item to the basket
        eng.addToBasketWithOptions(sessionId, 0, item1, basketOptions);

        /*
         * Retrieve the basket items from the engine. We need to save them and then read them back,
         * because the engine populates some attributes such as the encodedProduct. If we set
         * getStockReservationInfo to true in the AddToBasketOptions then the basket reserved stock
         * attributes are populated. These attributes are reservationId, qtyResrvdForResId,
         * reservationExpiryDate and reservationStartDate.
         */
        basketOptions.setGetStockReservationInfo(true);
        BasketIf[] items = eng.getBasketItemsPerCustomerWithOptions(sessionId, 0, DEFAULT_LANGUAGE,
                basketOptions);
        System.out
                .println("\ngetBasketItemsPerCustomerWithOptions()-The basket items with no stock or stock reservation information since stock hasn't been reserved yet.:\n");
        for (int i = 0; i < items.length; i++)
        {
            Basket b = (Basket) items[i];
            System.out.println(getBasketAttributes(b));
        }

        /*
         * Reserve the stock for the basket items. In the standard KonaKart storefront application
         * this is done in CheckoutAction.java just before displaying the checkout screen. Here we
         * set a reservation time of 30 seconds. If this isn't set then the global value from the
         * configuration variable STOCK_RESERVATION_TIME_SECS is used. The options can also be used
         * to limit the maximum number of reservations for a customer and to decide whether to allow
         * partial reservations if the quantity in stock is not sufficient.
         */
        StockReservationOptionsIf stockResOptions = new StockReservationOptions();
        stockResOptions.setReservationTimeSecs(30);
        items = eng.reserveStock(sessionId, items, stockResOptions);
        System.out
                .println("\nreserveStock() - The basket items with stock and stock reservation information:\n");
        for (int i = 0; i < items.length; i++)
        {
            Basket b = (Basket) items[i];
            System.out.println(getBasketAttributes(b));
        }

        /*
         * Get the basket items for the customer retrieving reserved stock information (e.g.
         * reservationId, qty reserved for this id, start and end dates of reservation). However
         * notice that overall stock levels are not returned. You must call
         * updateBasketWithStockInfoWithOptions() to get the system stock levels.
         */
        items = eng.getBasketItemsPerCustomerWithOptions(sessionId, 0, DEFAULT_LANGUAGE,
                basketOptions);
        System.out
                .println("\ngetBasketItemsPerCustomerWithOptions() - The basket items with stock reservation information:\n");
        for (int i = 0; i < items.length; i++)
        {
            Basket b = (Basket) items[i];
            System.out.println(getBasketAttributes(b));
        }

        /*
         * In order to get stock and reserved stock information we need to call the
         * updateBasketWithStockInfoWithOptions() API call
         */
        items = eng.updateBasketWithStockInfoWithOptions(items, basketOptions);
        System.out
                .println("\nupdateBasketWithStockInfoWithOptions() - The basket items with stock and stock reservation information:\n");
        for (int i = 0; i < items.length; i++)
        {
            Basket b = (Basket) items[i];
            System.out.println(getBasketAttributes(b));
        }

        /*
         * Remove the stock reservations
         */
        eng.removeStockReservationsForBasketItems(sessionId, items, null);

        /*
         * Retrieve the basket items for the customer which now don't have the stock reservation information.
         */
        items = eng.getBasketItemsPerCustomerWithOptions(sessionId, 0, DEFAULT_LANGUAGE,
                basketOptions);
        System.out
                .println("\ngetBasketItemsPerCustomerWithOptions() - The basket items with no stock reservation information since the reservations have been removed :\n");
        for (int i = 0; i < items.length; i++)
        {
            Basket b = (Basket) items[i];
            System.out.println(getBasketAttributes(b));
        }

    }

    /**
     * Method used to print out some Basket attributes
     * 
     * @param b
     * @return Returns a string with the attribute information
     */
    public String getBasketAttributes(Basket b)
    {
        StringBuffer str = new StringBuffer();
        str.append("Basket id : " + b.getId() + "\n");

        str.append("encodedProduct        = ").append(b.getEncodedProduct()).append("\n");
        str.append("quantity              = ").append(b.getQuantity()).append("\n");
        str.append("quantityInStock       = ").append(b.getQuantityInStock()).append("\n");
        str.append("quantityReserved      = ").append(b.getQuantityReserved()).append("\n");
        str.append("quantityAvailable     = ").append(b.getQuantityAvailable()).append("\n");
        str.append("qtyResrvdForResId     = ").append(b.getQtyResrvdForResId()).append("\n");
        str.append("reservationId         = ").append(b.getReservationId()).append("\n");
        if (b.getReservationStartDate() != null)
        {
            SimpleDateFormat sdf = new SimpleDateFormat("dd-MM-yy HH:mm.ss.SSS");
            str.append("reservationStartDate  = ")
                    .append(sdf.format(b.getReservationStartDate().getTime())).append("\n");
        } else
        {
            str.append("reservationStartDate  = ").append("null").append("\n");
        }
        if (b.getReservationExpiryDate() != null)
        {
            SimpleDateFormat sdf = new SimpleDateFormat("dd-MM-yy HH:mm.ss.SSS");
            str.append("reservationExpiryDate = ")
                    .append(sdf.format(b.getReservationExpiryDate().getTime())).append("\n");
        } else
        {
            str.append("reservationExpiryDate = ").append("null").append("\n");
        }

        return (str.toString());
    }
}
