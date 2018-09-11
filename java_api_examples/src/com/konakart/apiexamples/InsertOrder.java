//
// (c) 2006 DS Data Systems UK Ltd, All rights reserved.
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

import org.apache.commons.configuration.ConfigurationException;

import com.konakart.app.Basket;
import com.konakart.app.KKException;
import com.konakart.app.OrderStatusHistory;
import com.konakart.app.OrderUpdate;
import com.konakart.appif.BasketIf;
import com.konakart.appif.OptionIf;
import com.konakart.appif.OrderIf;
import com.konakart.appif.OrderStatusHistoryIf;
import com.konakart.appif.OrderUpdateIf;
import com.konakart.appif.PaymentDetailsIf;
import com.konakart.appif.ProductIf;
import com.konakart.appif.ShippingQuoteIf;

/**
 * This class shows how to call the KonaKart API to insert an order. Before running you may have to
 * edit BaseApiExample.java to change the username and password used to log into the engine. The
 * default values are doe@konakart.com / password .
 */
public class InsertOrder extends BaseApiExample
{
    private static final String usage = "Usage: InsertOrder\n" + COMMON_USAGE;

    // Matrox MG400-32MB in default DB. A product with options.
    private static int matroxProdId;

    // Blade Runner DVD in default DB.
    private static int bladeRunnerProdId;

    // Order status of Pending
    private static int PENDING = 1;

    // Order status of Processing
    private static int PROCESSING = 2;

    /**
     * @param args
     */
    public static void main(String[] args)
    {
        try
        {
            InsertOrder myInsertOrder = new InsertOrder();
            OrderIf order = myInsertOrder.insertOrder(args, usage);
            System.out.println("Order Inserted:\n" + order.toString());
        } catch (Exception e)
        {
            System.out.println("WARNING : There was a problem Inserting an Order");
            e.printStackTrace();
        }
    }

    /**
     * @param args
     *            command line arguments
     * @param use
     *            usage string
     * @return returns the inserted order object
     * @throws IllegalAccessException
     * @throws InstantiationException
     * @throws ClassNotFoundException
     * @throws KKException
     * @throws ConfigurationException
     * @throws IOException
     * @throws InvocationTargetException
     * @throws IllegalArgumentException
     */
    public OrderIf insertOrder(String[] args, String use) throws KKException,
            ClassNotFoundException, InstantiationException, IllegalAccessException,
            ConfigurationException, IOException, IllegalArgumentException,
            InvocationTargetException
    {
        parseArgs(args, use, 0);

        /*
         * Get an instance of the KonaKart engine and login. The method called can be found in
         * BaseApiExample.java
         */
        init();

        /*
         * Create an order. To do this we have to create an array of basketItems which we send to
         * the engine as a parameter of the createOrder() call. In this example, we will create a
         * basket item for a product that has options, and one for a product that doesn't have
         * options. The product ids that we choose, are the ones defined in the default database
         * delivered with the KonaKart installation package.
         */

        matroxProdId = getProductIdByName("Matrox G200 MMS");
        bladeRunnerProdId = getProductIdByName("Blade Runner - Director's Cut");

        // Just for diagnostics
        System.out.println("Matrox G200 MMS Product Id:               " + matroxProdId);
        System.out.println("Blade Runner - Director's Cut Product Id: " + bladeRunnerProdId);

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

        System.out.println("Product Read - Name: " + prod.getName());
        System.out.println("Product Read - Id:   " + prod.getId());

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
            eng.addToBasket(sessionId, 0, item);
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
        eng.addToBasket(sessionId, 0, item1);

        /*
         * Now that we have two different products in our basket, we can get the engine to create an
         * order for us.
         */
        // Retrieve the basket items from the engine. We need to save them and then read them
        // back, because the engine populates many attributes that are required to create the
        // order
        BasketIf[] items = eng.getBasketItemsPerCustomer(sessionId, 0, DEFAULT_LANGUAGE);

        // Get an order from the engine by passing it the basket items. Note that this order is
        // not yet saved to the DB since the shipping and payment information is missing.
        OrderIf order = eng.createOrder(sessionId, items, DEFAULT_LANGUAGE);

        /*
         * Normally we would get an order from the engine at the beginning of the checkout process.
         * As the customer chooses the shipping method and the payment type, more information is
         * added to the order which may affect the total order amount.
         */

        // Get shipping quotes for the order and choose the first one in the list for this order
        ShippingQuoteIf[] sQuotes = eng.getShippingQuotes(order, DEFAULT_LANGUAGE);
        if (sQuotes != null && sQuotes.length > 0)
        {
            order.setShippingQuote(sQuotes[0]);
        }

        // Get payment gateways / types available for the order and choose the first one in the
        // list for this order
        PaymentDetailsIf[] pGateways = eng.getPaymentGateways(order, DEFAULT_LANGUAGE);
        if (pGateways != null && pGateways.length > 0)
        {
            order.setPaymentDetails(pGateways[0]);
        }

        /*
         * Now that the order has been completed with all necessary information we can ask the
         * engine to calculate the order totals. We send it the order and receive back the same
         * order which includes order totals that the customer may check before confirming.
         */
        order = eng.getOrderTotals(order, DEFAULT_LANGUAGE);

        /*
         * We set the status of the order which it should have when first saved. In our default DB,
         * a status id == 1, means that the order is pending. i.e. it has been saved, but no other
         * action has been taken.
         */
        order.setStatus(PENDING);

        /*
         * The order has an array of OrderStatusHistory objects which track the various states that
         * an order may pass through, throughout its lifecycle. In all effects it keeps an audit
         * trail. We must set the first item in this array to say that it is in the pending state.
         * We may also add a comment such as "waiting for payment" to make it obvious what needs to
         * be done next.
         */
        OrderStatusHistoryIf status = new OrderStatusHistory();
        status.setOrderStatusId(PENDING);
        status.setUpdatedById(order.getCustomerId());
        status.setComments("Waiting for Payment");

        OrderStatusHistoryIf[] statusArray = new OrderStatusHistoryIf[]
        { status };
        order.setStatusTrail(statusArray);

        // The order has custom fields which may be set with custom information.
        order.setCustom1("custom1");
        order.setCustom5("custom5");

        // Now , finally the order can be saved.
        int orderId = eng.saveOrder(sessionId, order, DEFAULT_LANGUAGE);

        /*
         * Let's now assume that we have just received an instant payment notification from a
         * payment gateway to say that the order has been paid. When this happens we need to change
         * the state of the order to reflect the payment, and we may want to update the inventory to
         * reduce the stock level of the product that has been ordered.
         */

        // Change the status. This call changes the status and adds an element to the status
        // trail.

        OrderUpdateIf updateOrder = new OrderUpdate();
        updateOrder.setUpdatedById(order.getCustomerId());

        eng.updateOrder(sessionId, orderId, PROCESSING, /* Notify customer */false,
                "We've got the dosh, now let's ship it", updateOrder);

        // Update the inventory
        eng.updateInventory(sessionId, orderId);

        /*
         * At any point in time we can retrieve the order from the engine to view its details
         */
        order = eng.getOrder(sessionId, orderId, DEFAULT_LANGUAGE);

        System.out.println("Order " + order.getId() + " inserted successfully");

        return order;
    }
}
