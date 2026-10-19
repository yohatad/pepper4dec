// Sensor-liveness logic for bumper_guard. Time is passed in, so every case is
// exact rather than slept through.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "pepper_navigation/topic_liveness.h"

using pepper_navigation::TopicLiveness;

namespace
{
const std::vector<std::string> kTopics{"/points_safety", "/laser"};
TopicLiveness fresh() {return TopicLiveness(kTopics, 1.0);}
}  // namespace

TEST(TopicLiveness, NoRequiredTopicsMeansTheCheckIsOff)
{
  TopicLiveness l({}, 1.0);
  EXPECT_FALSE(l.watching());
  EXPECT_FALSE(l.any_stale(1e6));
}

TEST(TopicLiveness, NeverSeenCountsAsStale)
{
  auto l = fresh();
  EXPECT_TRUE(l.watching());
  // A source that never connected is exactly what this has to catch.
  EXPECT_TRUE(l.any_stale(0.0));
  EXPECT_EQ(l.stale_topics(0.0).size(), 2u);
  EXPECT_EQ(l.unseen().size(), 2u);
}

TEST(TopicLiveness, FreshTopicsAreNotStale)
{
  auto l = fresh();
  l.mark("/points_safety", 10.0);
  l.mark("/laser", 10.0);
  EXPECT_FALSE(l.any_stale(10.5));
  EXPECT_TRUE(l.unseen().empty());
}

TEST(TopicLiveness, GoesStaleAfterTheTimeout)
{
  auto l = fresh();
  l.mark("/points_safety", 10.0);
  l.mark("/laser", 10.0);
  EXPECT_FALSE(l.any_stale(11.0));    // boundary is exclusive
  EXPECT_TRUE(l.any_stale(11.1));
}

TEST(TopicLiveness, OneStaleTopicIsEnough)
{
  auto l = fresh();
  l.mark("/points_safety", 10.0);
  l.mark("/laser", 10.0);
  l.mark("/points_safety", 20.0);     // one keeps flowing, the other dies
  EXPECT_TRUE(l.any_stale(20.0));
  const auto bad = l.stale_topics(20.0);
  ASSERT_EQ(bad.size(), 1u);
  EXPECT_EQ(bad[0], "/laser");
}

TEST(TopicLiveness, RecoversWhenTheTopicReturns)
{
  auto l = fresh();
  l.mark("/points_safety", 10.0);
  l.mark("/laser", 10.0);
  ASSERT_TRUE(l.any_stale(12.0));
  l.mark("/points_safety", 12.0);
  l.mark("/laser", 12.0);
  EXPECT_FALSE(l.any_stale(12.0));
}

TEST(TopicLiveness, UnrequiredTopicIsIgnored)
{
  auto l = fresh();
  l.mark("/something_else", 10.0);    // not in the required list
  EXPECT_TRUE(l.any_stale(10.0));     // the two required ones are still unseen
  EXPECT_FALSE(l.stale("/something_else", 1e6));
}

TEST(TopicLiveness, ReportsAreStablyOrdered)
{
  auto l = fresh();
  const auto a = l.stale_topics(0.0);
  const auto b = l.stale_topics(0.0);
  EXPECT_EQ(a, b);
  EXPECT_EQ(a[0], "/laser");          // sorted, not insertion order
}
